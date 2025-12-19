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
        integrate::{Explicit, IntegrationError},
        optimize::{EqualityConstraint, FirstOrderRootFinding, OptimizationError},
    },
    mechanics::{DeformationGradientRateList, Times},
};

pub trait ViscoelasticFiniteElementBlock<C, F, const G: usize, const M: usize, const N: usize>
where
    C: Viscoelastic,
    F: ViscoelasticFiniteElement<C, G, M, N>,
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
        integrator: impl Explicit<NodalVelocities, NodalVelocitiesHistory>,
        time: &[Scalar],
        solver: impl FirstOrderRootFinding<NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>,
    ) -> Result<(Times, NodalCoordinatesHistory, NodalVelocitiesHistory), IntegrationError>;
    #[doc(hidden)]
    fn root_inner(
        &self,
        equality_constraint: &EqualityConstraint,
        nodal_coordinates: &NodalCoordinates,
        solver: &impl FirstOrderRootFinding<NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>,
        initial_guess: &NodalVelocities,
    ) -> Result<NodalVelocities, OptimizationError>;
}

impl<C, F, const G: usize, const M: usize, const N: usize>
    ViscoelasticFiniteElementBlock<C, F, G, M, N> for Block<C, F, N>
where
    C: Viscoelastic,
    F: ViscoelasticFiniteElement<C, G, M, N>,
    Self: SolidFiniteElementBlock<C, F, G, M, N>,
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
        integrator: impl Explicit<NodalVelocities, NodalVelocitiesHistory>,
        time: &[Scalar],
        solver: impl FirstOrderRootFinding<NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>,
    ) -> Result<(Times, NodalCoordinatesHistory, NodalVelocitiesHistory), IntegrationError> {
        let mut solution = NodalVelocities::zero(self.coordinates().len());
        integrator.integrate(
            |_: Scalar, nodal_coordinates: &NodalCoordinates| {
                solution =
                    self.root_inner(&equality_constraint, nodal_coordinates, &solver, &solution)?;
                Ok(solution.clone())
            },
            time,
            self.coordinates().clone().into(),
        )
    }
    fn root_inner(
        &self,
        equality_constraint: &EqualityConstraint,
        nodal_coordinates: &NodalCoordinates,
        solver: &impl FirstOrderRootFinding<NodalForcesSolid, NodalStiffnessesSolid, NodalVelocities>,
        initial_guess: &NodalVelocities,
    ) -> Result<NodalVelocities, OptimizationError> {
        solver.root(
            |nodal_velocities: &NodalVelocities| {
                Ok(self.nodal_forces(nodal_coordinates, nodal_velocities)?)
            },
            |nodal_velocities: &NodalVelocities| {
                Ok(self.nodal_stiffnesses(nodal_coordinates, nodal_velocities)?)
            },
            initial_guess.clone(),
            equality_constraint.clone(),
        )
    }
}
