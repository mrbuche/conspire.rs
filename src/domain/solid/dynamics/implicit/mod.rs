use super::{Solution, held};
use crate::{
    domain::{
        ElementModel, Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory,
        block::{element::Elements, finalize_node_neighbors, solver_from_neighbors},
        solid::{
            NodalForcesSolid, NodalStiffnessesSolid,
            elastic::ElasticElements,
            mass::{InertialMass, InverseMass},
        },
    },
    math::{
        Quantity, Tensor,
        integrate::{ImplicitDynamics, IntegrationError},
        optimize::{EqualityConstraints, RootFinding},
    },
    units::Time,
};

/// The motion of an elastic model with implicit time integration.
pub trait ElasticImplicitDynamics {
    /// Integrates the motion of the model with an implicit dynamics integrator.
    ///
    /// The masses may be lumped or consistent, and are added to the sparse tangent.
    /// Fixed degrees of freedom are held like the explicit integration does, and linear
    /// constraints are not supported.
    #[expect(clippy::too_many_arguments)]
    fn integrate(
        &self,
        integrator: &impl ImplicitDynamics<
            NodalCoordinates<3>,
            NodalForcesSolid<3>,
            NodalStiffnessesSolid<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        solver: impl RootFinding<NodalForcesSolid<3>, NodalStiffnessesSolid<3>, NodalCoordinates<3>>,
        time: &[Quantity<Time>],
        initial_state: (NodalCoordinates<3>, NodalVelocities<3>),
        external_forces: &NodalForcesSolid<3>,
        masses: &impl InertialMass<3>,
        equality_constraint: EqualityConstraints,
    ) -> Result<Solution, IntegrationError>;
}

impl<B> ElasticImplicitDynamics for Model<B, 3>
where
    B: ElasticElements<3>,
{
    fn integrate(
        &self,
        integrator: &impl ImplicitDynamics<
            NodalCoordinates<3>,
            NodalForcesSolid<3>,
            NodalStiffnessesSolid<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        solver: impl RootFinding<NodalForcesSolid<3>, NodalStiffnessesSolid<3>, NodalCoordinates<3>>,
        time: &[Quantity<Time>],
        (initial_coordinates, initial_velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        external_forces: &NodalForcesSolid<3>,
        masses: &impl InertialMass<3>,
        equality_constraint: EqualityConstraints,
    ) -> Result<Solution, IntegrationError> {
        let (inverse, initial_velocities) =
            held(masses, equality_constraint.clone(), initial_velocities)?;
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let sparse = solver_from_neighbors(&neighbors, &equality_constraint, 3, false);
        integrator.integrate(
            |_, coordinates: &NodalCoordinates<3>, _: &NodalVelocities<3>| {
                Ok(inverse.nodal_accelerations(
                    external_forces,
                    &self
                        .nodal_forces(coordinates)
                        .map_err(|error| error.to_string())?,
                ))
            },
            |_,
             coordinates: &NodalCoordinates<3>,
             _: &NodalVelocities<3>,
             accelerations: &NodalAccelerations<3>| {
                Ok(InertialMass::inertial_forces(masses, accelerations)
                    + self
                        .nodal_forces(coordinates)
                        .map_err(|error| error.to_string())?
                    - external_forces)
            },
            |_, coordinates: &NodalCoordinates<3>, _: &NodalVelocities<3>, dt, beta, _| {
                let mut tangent = self
                    .nodal_stiffnesses(coordinates)
                    .map_err(|error| error.to_string())?;
                masses.add_to_stiffnesses(1.0 / (beta * dt.value().powi(2)), &mut tangent);
                Ok(tangent)
            },
            solver,
            equality_constraint,
            Some(sparse),
            time,
            initial_coordinates,
            initial_velocities,
        )
    }
}
