//! Whether a run of explicit dynamics checks that its time step is stable.

#[cfg(all(test, feature = "cbm"))]
mod test;

use super::{Lumped, MassChoice};
use crate::{
    domain::{
        Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory,
        solid::{dynamics::Solution, mass::LumpedMassElements, time_scale::TimeScaleElements},
    },
    math::{
        Quantity, Scalar,
        integrate::{ExplicitDynamics, IntegrationError},
    },
    units::Time,
};

/// Whether a run of explicit dynamics checks that its time step is stable, for a choice of
/// masses, which a policy that checks may not support.
pub trait Stability<M, C>
where
    C: MassChoice<M>,
{
    /// Takes the steps, with the accelerations from `function`, and `held` gives the coordinates
    /// as the motion that holds some of them has them.
    fn integrate(
        self,
        model: &M,
        held: impl Fn(Quantity<Time>, &NodalCoordinates<3>) -> NodalCoordinates<3>,
        integrator: &impl ExplicitDynamics<
            NodalCoordinates<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        function: impl FnMut(
            Quantity<Time>,
            &NodalCoordinates<3>,
            &NodalVelocities<3>,
        ) -> Result<NodalAccelerations<3>, String>,
        time: &[Quantity<Time>],
        initial_state: (NodalCoordinates<3>, NodalVelocities<3>),
    ) -> Result<Solution, IntegrationError>;
}

/// Takes the steps that are given, however large they are.
#[derive(Clone, Copy, Debug, Default)]
pub struct Unchecked;

impl<M, C> Stability<M, C> for Unchecked
where
    C: MassChoice<M>,
{
    fn integrate(
        self,
        _: &M,
        _: impl Fn(Quantity<Time>, &NodalCoordinates<3>) -> NodalCoordinates<3>,
        integrator: &impl ExplicitDynamics<
            NodalCoordinates<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        function: impl FnMut(
            Quantity<Time>,
            &NodalCoordinates<3>,
            &NodalVelocities<3>,
        ) -> Result<NodalAccelerations<3>, String>,
        time: &[Quantity<Time>],
        initial_state: (NodalCoordinates<3>, NodalVelocities<3>),
    ) -> Result<Solution, IntegrationError> {
        let (initial_coordinates, initial_velocities) = initial_state;
        integrator.integrate(function, time, initial_coordinates, initial_velocities)
    }
}

/// Checks each step against the fastest time scale of the elements, and a step above the
/// fraction `safety` of the stability limit is an error.
///
/// The estimate assembles the element stiffnesses, and is refreshed every `interval`
/// evaluations of the bound, which for the integrators here is every step. It is that of the
/// lumped masses, which a consistent mass exceeds, so only [`Lumped`] is supported.
#[derive(Clone, Copy, Debug)]
pub struct Checked {
    pub safety: Scalar,
    pub interval: usize,
}

impl<B> Stability<Model<B, 3>, Lumped> for Checked
where
    B: LumpedMassElements<3> + TimeScaleElements<3>,
{
    fn integrate(
        self,
        model: &Model<B, 3>,
        held: impl Fn(Quantity<Time>, &NodalCoordinates<3>) -> NodalCoordinates<3>,
        integrator: &impl ExplicitDynamics<
            NodalCoordinates<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        function: impl FnMut(
            Quantity<Time>,
            &NodalCoordinates<3>,
            &NodalVelocities<3>,
        ) -> Result<NodalAccelerations<3>, String>,
        time: &[Quantity<Time>],
        initial_state: (NodalCoordinates<3>, NodalVelocities<3>),
    ) -> Result<Solution, IntegrationError> {
        let Self { safety, interval } = self;
        let (initial_coordinates, initial_velocities) = initial_state;
        if interval == 0 {
            return Err(IntegrationError::Intermediate(
                "The interval between estimates of the time scale must be at least one."
                    .to_string(),
            ));
        }
        let mut evaluations = 0;
        let mut time_scale = Time::seconds(Scalar::INFINITY);
        integrator.integrate_bounded(
            function,
            |time, coordinates: &NodalCoordinates<3>, _: &NodalVelocities<3>| {
                if evaluations % interval == 0 {
                    time_scale = model
                        .fastest_time_scale(&model.coordinates, &held(time, coordinates))
                        .map_err(|error| error.to_string())?;
                }
                evaluations += 1;
                Ok(time_scale)
            },
            safety,
            time,
            initial_coordinates,
            initial_velocities,
        )
    }
}
