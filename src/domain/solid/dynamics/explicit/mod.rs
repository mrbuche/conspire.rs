use super::{Solution, motion::Motion};
use crate::{
    domain::{
        ElementModelError, Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory, Stepping,
        boundary::{ConstrainAt, LoadAt, Solid},
        solid::{
            NodalForcesSolid,
            elastic::{ElasticElements, ParallelElasticElements},
            mass::InverseMass,
        },
    },
    math::{
        Quantity,
        integrate::{ExplicitDynamics, IntegrationError},
    },
    units::Time,
};

#[cfg(feature = "gpu")]
mod device;
mod execution;
mod stability;

pub use crate::domain::solid::mass::{Consistent, Lumped, MassChoice};
#[cfg(feature = "gpu")]
pub use device::DeviceExplicitDynamics;
pub use execution::Execution;
pub use stability::{Checked, Stability, Unchecked};

/// The motion of an elastic model with explicit time integration.
pub trait ElasticExplicitDynamics
where
    Self: Sized,
{
    fn nodal_accelerations(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        time: Quantity<Time>,
        applied_loads: &impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
        masses: &impl InverseMass<3>,
    ) -> Result<NodalAccelerations<3>, ElementModelError>;
    /// Integrates the motion of the model with an explicit dynamics integrator.
    ///
    /// The masses are those of `mass` ([`Lumped`] or [`Consistent`]) that the blocks of the model
    /// make, and `stability` ([`Unchecked`] or [`Checked`]) says whether the time step is checked
    /// against the stability limit, which a policy may only do for some choices of masses. Fixed
    /// degrees of freedom, whose indices are `D * node + component`, are held by giving them no
    /// velocity, and no acceleration from the inverse of the masses. Linear constraints are not
    /// supported. The nodal forces are computed as `execution` says, with identical results
    /// either way.
    #[expect(clippy::too_many_arguments)]
    fn integrate<C>(
        &self,
        execution: Execution,
        stability: impl Stability<Self, C>,
        mass: C,
        integrator: &impl ExplicitDynamics<
            NodalCoordinates<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        time: &[Quantity<Time>],
        initial_state: (NodalCoordinates<3>, NodalVelocities<3>),
        applied_loads: &impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
        conditions: &impl ConstrainAt<Solid<3>>,
    ) -> Result<Solution, IntegrationError>
    where
        C: MassChoice<Self>;
}

impl<B> ElasticExplicitDynamics for Model<B, 3>
where
    B: ParallelElasticElements<3>,
{
    fn nodal_accelerations(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        time: Quantity<Time>,
        applied_loads: &impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
        masses: &impl InverseMass<3>,
    ) -> Result<NodalAccelerations<3>, ElementModelError> {
        let mut unbalanced_forces = self.nodal_forces(nodal_coordinates)?;
        applied_loads
            .subtract_at(time, nodal_coordinates, &mut unbalanced_forces)
            .map_err(|error| ElementModelError::upstream(error, "the applied loads"))?;
        Ok(masses.nodal_accelerations(&unbalanced_forces))
    }
    fn integrate<C>(
        &self,
        execution: Execution,
        stability: impl Stability<Self, C>,
        mass: C,
        integrator: &impl ExplicitDynamics<
            NodalCoordinates<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        time: &[Quantity<Time>],
        (mut initial_coordinates, initial_velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        applied_loads: &impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
        conditions: &impl ConstrainAt<Solid<3>>,
    ) -> Result<Solution, IntegrationError>
    where
        C: MassChoice<Self>,
    {
        let initial_time = *time
            .first()
            .ok_or(IntegrationError::LengthTimeLessThanTwo)?;
        let masses = mass.masses(self);
        let (masses, initial_velocities, motion) = Motion::held(
            &masses,
            conditions,
            initial_time,
            &mut initial_coordinates,
            initial_velocities,
            Stepping::Explicit,
        )?;
        let mut run = execution.start(self)?;
        let (times, mut coordinates, velocities, accelerations) = stability.integrate(
            self,
            |time, coordinates| {
                let mut coordinates = coordinates.clone();
                motion.apply(time, &mut coordinates);
                coordinates
            },
            integrator,
            |time, coordinates: &NodalCoordinates<3>, _: &NodalVelocities<3>| {
                let mut coordinates = coordinates.clone();
                motion.apply(time, &mut coordinates);
                self.nodal_accelerations_in(&mut run, &coordinates, time, applied_loads, &masses)
                    .map_err(|error| error.to_string())
            },
            time,
            (initial_coordinates, initial_velocities),
        )?;
        motion.record(&times, &mut coordinates);
        Ok((times, coordinates, velocities, accelerations))
    }
}
