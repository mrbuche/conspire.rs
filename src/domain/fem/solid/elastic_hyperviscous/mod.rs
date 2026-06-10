use crate::{
    fem::{
        Blocks, FiniteElementModel, FiniteElementModelError, FiniteElements, Model,
        NodalCoordinates, NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory,
        block::{band_from_neighbors, finalize_node_neighbors},
        solid::{
            NodalForcesSolid, NodalStiffnessesSolid, viscoelastic::ViscoelasticFiniteElements,
        },
    },
    math::{
        Scalar, Tensor,
        integrate::{ImplicitDaeSecondOrderMinimize, IntegrationError},
        optimize::{EqualityConstraint, SecondOrderOptimization},
    },
    mechanics::Times,
};

pub trait ElasticHyperviscousFiniteElements
where
    Self: ViscoelasticFiniteElements,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<Scalar, FiniteElementModelError>;
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<Scalar, FiniteElementModelError>;
}

impl<B> ElasticHyperviscousFiniteElements for Model<B>
where
    B: ElasticHyperviscousFiniteElements,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<Scalar, FiniteElementModelError> {
        self.blocks
            .viscous_dissipation(nodal_coordinates, nodal_velocities)
    }
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<Scalar, FiniteElementModelError> {
        self.blocks
            .dissipation_potential(nodal_coordinates, nodal_velocities)
    }
}

impl<B1, B2> ElasticHyperviscousFiniteElements for Blocks<B1, B2>
where
    B1: ElasticHyperviscousFiniteElements,
    B2: ElasticHyperviscousFiniteElements,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<Scalar, FiniteElementModelError> {
        Ok(self
            .0
            .viscous_dissipation(nodal_coordinates, nodal_velocities)?
            + self
                .1
                .viscous_dissipation(nodal_coordinates, nodal_velocities)?)
    }
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<Scalar, FiniteElementModelError> {
        Ok(self
            .0
            .dissipation_potential(nodal_coordinates, nodal_velocities)?
            + self
                .1
                .dissipation_potential(nodal_coordinates, nodal_velocities)?)
    }
}

pub trait SecondOrderMinimize {
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

impl<B> SecondOrderMinimize for Model<B>
where
    B: ElasticHyperviscousFiniteElements,
{
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
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let banded = band_from_neighbors(&neighbors, &equality_constraint, 3);
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
