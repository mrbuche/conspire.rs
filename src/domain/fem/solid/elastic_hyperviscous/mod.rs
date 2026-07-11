use crate::{
    fem::{
        Blocks, ElementModel, ElementModelError, Elements, Model, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory,
        block::{band_from_neighbors, finalize_node_neighbors},
        solid::{NodalForcesSolid, NodalStiffnessesSolid, viscoelastic::ViscoelasticElements},
    },
    math::{
        Scalar, Tensor,
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
    ) -> Result<Scalar, ElementModelError>;
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<Scalar, ElementModelError>;
}

impl<B, const D: usize> ElasticHyperviscousElements<D> for Model<B, D>
where
    B: ElasticHyperviscousElements<D>,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<Scalar, ElementModelError> {
        self.blocks
            .viscous_dissipation(nodal_coordinates, nodal_velocities)
    }
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<Scalar, ElementModelError> {
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
    ) -> Result<Scalar, ElementModelError> {
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
    ) -> Result<Scalar, ElementModelError> {
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
            NodalStiffnessesSolid<D>,
            NodalVelocities<D>,
            NodalVelocitiesHistory<D>,
        >,
        time: &[Scalar],
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
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
            NodalStiffnessesSolid<D>,
            NodalVelocities<D>,
            NodalVelocitiesHistory<D>,
        >,
        time: &[Scalar],
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
    ) -> Result<(Times, NodalCoordinatesHistory<D>, NodalVelocitiesHistory<D>), IntegrationError>
    {
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let banded = band_from_neighbors(&neighbors, &equality_constraint, D);
        integrator.integrate(
            |_: Scalar,
             nodal_coordinates: &NodalCoordinates<D>,
             nodal_velocities: &NodalVelocities<D>| {
                Ok(self.dissipation_potential(nodal_coordinates, nodal_velocities)?)
            },
            |_: Scalar,
             nodal_coordinates: &NodalCoordinates<D>,
             nodal_velocities: &NodalVelocities<D>| {
                Ok(self.nodal_forces(nodal_coordinates, nodal_velocities)?)
            },
            |_: Scalar,
             nodal_coordinates: &NodalCoordinates<D>,
             nodal_velocities: &NodalVelocities<D>| {
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
