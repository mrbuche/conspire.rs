use crate::{
    constitutive::solid::viscoelastic::Viscoelastic,
    fem::{
        NodalCoordinates, NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory,
        block::{
            Block, FiniteElementBlockError,
            element::{
                ElementNodalVelocities, FiniteElementError,
                solid::viscoelastic::ViscoelasticFiniteElement,
            },
            solid::{NodalForcesSolid, NodalStiffnessesSolid, SolidFiniteElementBlock},
        },
    },
    math::{
        Scalar, Tensor,
        integrate::{ImplicitDaeFirstOrderRoot, IntegrationError},
        optimize::{EqualityConstraint, FirstOrderRootFinding},
    },
    mechanics::{DeformationGradientRateList, Times},
};

pub trait ViscoelasticFiniteElementBlock<
    C,
    F,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    C: Viscoelastic,
    F: ViscoelasticFiniteElement<C, G, M, N, P>,
{
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Vec<DeformationGradientRateList<G>>;
    fn element_velocities(
        &self,
        nodal_velocities: &NodalVelocities,
        nodes: &[usize; N],
    ) -> ElementNodalVelocities<N>;
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<NodalForcesSolid, FiniteElementBlockError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<NodalStiffnessesSolid, FiniteElementBlockError>;
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        integrator: impl ImplicitDaeFirstOrderRoot<
            NodalForcesSolid,
            NodalStiffnessesSolid,
            NodalVelocities,
            NodalVelocitiesHistory,
        >,
        time: &[Scalar],
        solver: impl FirstOrderRootFinding<NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>,
    ) -> Result<(Times, NodalCoordinatesHistory, NodalVelocitiesHistory), IntegrationError>;
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    ViscoelasticFiniteElementBlock<C, F, G, M, N, P> for Block<C, F, G, M, N, P>
where
    C: Viscoelastic,
    F: ViscoelasticFiniteElement<C, G, M, N, P>,
    Self: SolidFiniteElementBlock<C, F, G, M, N, P>,
{
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Vec<DeformationGradientRateList<G>> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, nodes)| {
                element.deformation_gradient_rates(
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    &self.element_velocities(nodal_velocities, nodes),
                )
            })
            .collect()
    }
    fn element_velocities(
        &self,
        velocities: &NodalVelocities,
        nodes: &[usize; N],
    ) -> ElementNodalVelocities<N> {
        nodes.iter().map(|&node| velocities[node].clone()).collect()
    }
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<NodalForcesSolid, FiniteElementBlockError> {
        let mut nodal_forces = NodalForcesSolid::zero(nodal_coordinates.len());
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                        &self.element_velocities(nodal_velocities, nodes),
                    )?
                    .iter()
                    .zip(nodes)
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), FiniteElementError>(())
            }) {
            Ok(()) => Ok(nodal_forces),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<NodalStiffnessesSolid, FiniteElementBlockError> {
        let mut nodal_stiffnesses = NodalStiffnessesSolid::zero(nodal_coordinates.len());
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                        &self.element_velocities(nodal_velocities, nodes),
                    )?
                    .iter()
                    .zip(nodes)
                    .for_each(|(object, &node_a)| {
                        object
                            .iter()
                            .zip(nodes)
                            .for_each(|(nodal_stiffness, &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                            })
                    });
                Ok::<(), FiniteElementError>(())
            }) {
            Ok(()) => Ok(nodal_stiffnesses),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        integrator: impl ImplicitDaeFirstOrderRoot<
            NodalForcesSolid,
            NodalStiffnessesSolid,
            NodalVelocities,
            NodalVelocitiesHistory,
        >,
        time: &[Scalar],
        solver: impl FirstOrderRootFinding<NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>,
    ) -> Result<(Times, NodalCoordinatesHistory, NodalVelocitiesHistory), IntegrationError> {
        integrator.integrate(
            |_: Scalar,
             nodal_coordinates: &NodalCoordinates,
             nodal_velocities: &NodalVelocities| {
                Ok(self.nodal_forces(nodal_coordinates, nodal_velocities)?)
            },
            |_: Scalar,
             nodal_coordinates: &NodalCoordinates,
             nodal_velocities: &NodalVelocities| {
                Ok(self.nodal_stiffnesses(nodal_coordinates, nodal_velocities)?)
            },
            solver,
            time,
            (
                self.coordinates().clone().into(),
                NodalVelocities::zero(self.coordinates().len()),
            ),
            |_: Scalar| equality_constraint.clone(),
        )
    }
}
