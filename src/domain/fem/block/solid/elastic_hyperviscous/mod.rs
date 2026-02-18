use crate::{
    constitutive::solid::elastic_hyperviscous::ElasticHyperviscous,
    fem::{
        NodalCoordinates, NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory,
        block::{
            Block, FiniteElementBlockError, band,
            element::solid::elastic_hyperviscous::ElasticHyperviscousFiniteElement,
            solid::{
                NodalForcesSolid, NodalStiffnessesSolid,
                viscoelastic::ViscoelasticFiniteElementBlock,
            },
        },
    },
    math::{
        Scalar, Tensor,
        integrate::{ImplicitDaeSecondOrderMinimize, IntegrationError},
        optimize::{EqualityConstraint, SecondOrderOptimization},
    },
    mechanics::Times,
};

pub trait ElasticHyperviscousFiniteElementBlock<
    C,
    F,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    C: ElasticHyperviscous,
    F: ElasticHyperviscousFiniteElement<C, G, M, N, P>,
    Self: ViscoelasticFiniteElementBlock<C, F, G, M, N, P>,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<Scalar, FiniteElementBlockError>;
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<Scalar, FiniteElementBlockError>;
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        integrator: impl ImplicitDaeSecondOrderMinimize<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            NodalVelocities,
            NodalVelocitiesHistory,
        >,
        time: &[Scalar],
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            NodalCoordinates,
        >,
    ) -> Result<(Times, NodalCoordinatesHistory, NodalVelocitiesHistory), IntegrationError>;
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    ElasticHyperviscousFiniteElementBlock<C, F, G, M, N, P> for Block<C, F, G, M, N, P>
where
    C: ElasticHyperviscous,
    F: ElasticHyperviscousFiniteElement<C, G, M, N, P>,
    Self: ViscoelasticFiniteElementBlock<C, F, G, M, N, P>,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<Scalar, FiniteElementBlockError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, nodes)| {
                element.viscous_dissipation(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    &self.element_velocities(nodal_velocities, nodes),
                )
            })
            .sum()
        {
            Ok(viscous_dissipation) => Ok(viscous_dissipation),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<Scalar, FiniteElementBlockError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, nodes)| {
                element.dissipation_potential(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    &self.element_velocities(nodal_velocities, nodes),
                )
            })
            .sum()
        {
            Ok(dissipation_potential) => Ok(dissipation_potential),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        integrator: impl ImplicitDaeSecondOrderMinimize<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            NodalVelocities,
            NodalVelocitiesHistory,
        >,
        time: &[Scalar],
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            NodalCoordinates,
        >,
    ) -> Result<(Times, NodalCoordinatesHistory, NodalVelocitiesHistory), IntegrationError> {
        let banded = band(
            self.connectivity(),
            &equality_constraint,
            self.coordinates().len(),
            3,
        );
        integrator.integrate(
            |_: Scalar,
             nodal_coordinates: &NodalCoordinates,
             nodal_velocities: &NodalVelocities| {
                Ok(self.dissipation_potential(nodal_coordinates, nodal_velocities)?)
            },
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
            self.coordinates().clone().into(),
            |_: Scalar| equality_constraint.clone(),
            Some(banded),
        )
    }
}
