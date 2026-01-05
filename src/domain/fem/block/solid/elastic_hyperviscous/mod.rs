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
        integrate::{Explicit, IntegrationError},
        optimize::{EqualityConstraint, OptimizationError, SecondOrderOptimization},
    },
    mechanics::Times,
};

pub trait ElasticHyperviscousFiniteElementBlock<
    C,
    F,
    const G: usize,
    const M: usize,
    const N: usize,
> where
    C: ElasticHyperviscous,
    F: ElasticHyperviscousFiniteElement<C, G, M, N>,
    Self: ViscoelasticFiniteElementBlock<C, F, G, M, N>,
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
        integrator: impl Explicit<NodalVelocities, NodalVelocitiesHistory>,
        time: &[Scalar],
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            NodalCoordinates,
        >,
    ) -> Result<(Times, NodalCoordinatesHistory, NodalVelocitiesHistory), IntegrationError>;
    #[doc(hidden)]
    fn minimize_inner(
        &self,
        equality_constraint: &EqualityConstraint,
        nodal_coordinates: &NodalCoordinates,
        solver: &impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            NodalCoordinates,
        >,
        initial_guess: &NodalVelocities,
    ) -> Result<NodalVelocities, OptimizationError>;
}

impl<C, F, const G: usize, const M: usize, const N: usize>
    ElasticHyperviscousFiniteElementBlock<C, F, G, M, N> for Block<C, F, G, M, N>
where
    C: ElasticHyperviscous,
    F: ElasticHyperviscousFiniteElement<C, G, M, N>,
    Self: ViscoelasticFiniteElementBlock<C, F, G, M, N>,
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
        integrator: impl Explicit<NodalVelocities, NodalVelocitiesHistory>,
        time: &[Scalar],
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            NodalCoordinates,
        >,
    ) -> Result<(Times, NodalCoordinatesHistory, NodalVelocitiesHistory), IntegrationError> {
        let mut solution = NodalVelocities::zero(self.coordinates().len());
        integrator.integrate(
            |_: Scalar, nodal_coordinates: &NodalCoordinates| {
                solution = self.minimize_inner(
                    &equality_constraint,
                    nodal_coordinates,
                    &solver,
                    &solution,
                )?;
                Ok(solution.clone())
            },
            time,
            self.coordinates().clone().into(),
        )
    }
    fn minimize_inner(
        &self,
        equality_constraint: &EqualityConstraint,
        nodal_coordinates: &NodalCoordinates,
        solver: &impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            NodalVelocities,
        >,
        initial_guess: &NodalVelocities,
    ) -> Result<NodalVelocities, OptimizationError> {
        let banded = band(
            self.connectivity(),
            equality_constraint,
            nodal_coordinates.len(),
            3,
        );
        solver.minimize(
            |nodal_velocities: &NodalVelocities| {
                Ok(self.dissipation_potential(nodal_coordinates, nodal_velocities)?)
            },
            |nodal_velocities: &NodalVelocities| {
                Ok(self.nodal_forces(nodal_coordinates, nodal_velocities)?)
            },
            |nodal_velocities: &NodalVelocities| {
                Ok(self.nodal_stiffnesses(nodal_coordinates, nodal_velocities)?)
            },
            initial_guess.clone(),
            equality_constraint.clone(),
            Some(banded),
        )
    }
}
