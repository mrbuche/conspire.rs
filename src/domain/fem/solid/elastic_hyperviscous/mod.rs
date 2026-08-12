use crate::{
    fem::{
        Blocks, ElementModel, ElementModelError, Elements, Model, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory,
        block::{finalize_node_neighbors, solver_from_neighbors},
        solid::{NodalDampingsSolid, NodalForcesSolid, viscoelastic::ViscoelasticElements},
    },
    math::{
        Dissipation, Quantity, Scalar, Tensor, Time,
        integrate::{ImplicitDaeSecondOrderMinimize, IntegrationError},
        optimize::{EqualityConstraint, SecondOrderOptimization},
    },
    mechanics::Times,
};

pub trait ElasticHyperviscousElements<const D: usize>
where
    Self: ViscoelasticElements<D>,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<Quantity<Dissipation>, ElementModelError>;
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<Quantity<Dissipation>, ElementModelError>;
}

impl<B, const D: usize> ElasticHyperviscousElements<D> for Model<B, D>
where
    B: ElasticHyperviscousElements<D>,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<Quantity<Dissipation>, ElementModelError> {
        self.blocks
            .viscous_dissipation(nodal_coordinates, nodal_velocities)
    }
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<Quantity<Dissipation>, ElementModelError> {
        self.blocks
            .dissipation_potential(nodal_coordinates, nodal_velocities)
    }
}

impl<B1, B2, const D: usize> ElasticHyperviscousElements<D> for Blocks<B1, B2>
where
    B1: ElasticHyperviscousElements<D>,
    B2: ElasticHyperviscousElements<D>,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<Quantity<Dissipation>, ElementModelError> {
        Ok(self
            .0
            .viscous_dissipation(nodal_coordinates, nodal_velocities)?
            + self
                .1
                .viscous_dissipation(nodal_coordinates, nodal_velocities)?)
    }
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<Quantity<Dissipation>, ElementModelError> {
        Ok(self
            .0
            .dissipation_potential(nodal_coordinates, nodal_velocities)?
            + self
                .1
                .dissipation_potential(nodal_coordinates, nodal_velocities)?)
    }
}

pub trait SecondOrderMinimize<const D: usize> {
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        integrator: impl ImplicitDaeSecondOrderMinimize<
            Scalar,
            NodalForcesSolid<D>,
            NodalDampingsSolid<D>,
            NodalCoordinates<D>,
            NodalCoordinatesHistory<D>,
            NodalVelocitiesHistory<D>,
        >,
        time: &[Quantity<Time>],
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid<D>,
            NodalDampingsSolid<D>,
            NodalVelocities<D>,
        >,
    ) -> Result<(Times, NodalCoordinatesHistory<D>, NodalVelocitiesHistory<D>), IntegrationError>;
}

impl<B, const D: usize> SecondOrderMinimize<D> for Model<B, D>
where
    B: ElasticHyperviscousElements<D>,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        integrator: impl ImplicitDaeSecondOrderMinimize<
            Scalar,
            NodalForcesSolid<D>,
            NodalDampingsSolid<D>,
            NodalCoordinates<D>,
            NodalCoordinatesHistory<D>,
            NodalVelocitiesHistory<D>,
        >,
        time: &[Quantity<Time>],
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid<D>,
            NodalDampingsSolid<D>,
            NodalVelocities<D>,
        >,
    ) -> Result<(Times, NodalCoordinatesHistory<D>, NodalVelocitiesHistory<D>), IntegrationError>
    {
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let sparse = solver_from_neighbors(&neighbors, &equality_constraint, D, true);
        integrator.integrate(
            |_: Quantity<Time>,
             nodal_coordinates: &NodalCoordinates<D>,
             nodal_velocities: &NodalVelocities<D>| {
                // The solver only compares its objective, so the dissipation
                // is spent where it is handed over.
                Ok(self
                    .dissipation_potential(nodal_coordinates, nodal_velocities)?
                    .value_as::<Dissipation>())
            },
            |_: Quantity<Time>,
             nodal_coordinates: &NodalCoordinates<D>,
             nodal_velocities: &NodalVelocities<D>| {
                Ok(self.nodal_forces(nodal_coordinates, nodal_velocities)?)
            },
            |_: Quantity<Time>,
             nodal_coordinates: &NodalCoordinates<D>,
             nodal_velocities: &NodalVelocities<D>| {
                Ok(self.nodal_stiffnesses(nodal_coordinates, nodal_velocities)?)
            },
            solver,
            time,
            self.coordinates().clone().into(),
            |_: Quantity<Time>| equality_constraint.clone(),
            Some(sparse),
        )
    }
}
