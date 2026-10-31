use super::{Solution, motion::Motion};
use crate::{
    domain::{
        ElementModelError, Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory, Stepping,
        boundary::{ConstrainAt, LoadAt, Solid},
        solid::{
            NodalForcesSolid,
            elastic::{ElasticElements, ParallelElasticElements, Workers},
            mass::{InverseMass, MassMatrix, NodalLumpedMasses},
            time_scale::TimeScaleElements,
        },
    },
    math::{
        Quantity, Scalar, Tensor,
        integrate::{ExplicitDynamics, IntegrationError},
    },
    units::Time,
};
use std::thread::available_parallelism;

/// How the nodal forces of a run of explicit dynamics are computed.
///
/// The choice never changes the results, which are identical in either case, and only the run
/// is affected: the threads and their storage are created when the run starts and dropped
/// when it ends.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Execution {
    /// On the calling thread, with nothing created for the run.
    #[default]
    Serial,
    /// On the given number of threads, which live for the run. Any number, including one,
    /// takes the parallel pathway, which [`Serial`](Self::Serial) avoids, and zero is as many
    /// threads as the machine offers.
    Threads(usize),
}

/// The threads and storage of one run.
enum Run<S> {
    Serial,
    Parallel(Workers, S),
}

impl Execution {
    fn start<B>(self, model: &Model<B, 3>) -> Result<Run<B::Scratch>, IntegrationError>
    where
        B: ParallelElasticElements<3>,
    {
        let threads = match self {
            Self::Serial => return Ok(Run::Serial),
            Self::Threads(0) => available_parallelism().map_or(1, usize::from),
            Self::Threads(threads) => threads,
        };
        Ok(Run::Parallel(
            Workers::new(threads).map_err(IntegrationError::Intermediate)?,
            model.scratch(),
        ))
    }
}

impl<B> Model<B, 3>
where
    B: ParallelElasticElements<3>,
{
    fn nodal_accelerations_in(
        &self,
        run: &mut Run<B::Scratch>,
        nodal_coordinates: &NodalCoordinates<3>,
        time: Quantity<Time>,
        applied_loads: &impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
        masses: &impl InverseMass<3>,
    ) -> Result<NodalAccelerations<3>, ElementModelError> {
        match run {
            Run::Serial => self.nodal_accelerations(nodal_coordinates, time, applied_loads, masses),
            Run::Parallel(workers, scratch) => {
                let mut unbalanced_forces = NodalForcesSolid::zero(nodal_coordinates.len());
                self.nodal_forces_parallel_into(
                    scratch,
                    workers,
                    nodal_coordinates,
                    &mut unbalanced_forces,
                )?;
                applied_loads
                    .subtract_at(time, nodal_coordinates, &mut unbalanced_forces)
                    .map_err(|error| ElementModelError::upstream(error, "the applied loads"))?;
                Ok(masses.nodal_accelerations(&unbalanced_forces))
            }
        }
    }
}

/// The motion of an elastic model with explicit time integration.
pub trait ElasticExplicitDynamics {
    fn nodal_accelerations(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        time: Quantity<Time>,
        applied_loads: &impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
        masses: &impl InverseMass<3>,
    ) -> Result<NodalAccelerations<3>, ElementModelError>;
    /// Integrates the motion of the model with an explicit dynamics integrator.
    ///
    /// The masses may be lumped or consistent. Fixed degrees of freedom, whose indices are
    /// `D * node + component`, are held by giving them no velocity, and no acceleration
    /// from the inverse of the masses. Linear constraints are not supported.
    #[expect(clippy::too_many_arguments)]
    fn integrate(
        &self,
        execution: Execution,
        integrator: &impl ExplicitDynamics<
            NodalCoordinates<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        time: &[Quantity<Time>],
        initial_state: (NodalCoordinates<3>, NodalVelocities<3>),
        applied_loads: &impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
        masses: &impl MassMatrix<3>,
        conditions: &impl ConstrainAt<Solid<3>>,
    ) -> Result<Solution, IntegrationError>;
    /// Integrates the motion of the model like [`integrate`](Self::integrate), with a time
    /// step limited by the fastest time scale of the elements.
    ///
    /// Only lumped masses are accepted, since the estimate of the highest frequency is that of
    /// the lumped masses, which a consistent mass exceeds. The estimate assembles the element
    /// stiffnesses, and is refreshed every `interval` evaluations of the bound, which for the
    /// integrators here is every step. The time step may use at most the fraction `safety` of
    /// the stability limit, and a step above it is an error.
    #[expect(clippy::too_many_arguments)]
    fn integrate_bounded(
        &self,
        execution: Execution,
        integrator: &impl ExplicitDynamics<
            NodalCoordinates<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        safety: Scalar,
        interval: usize,
        time: &[Quantity<Time>],
        initial_state: (NodalCoordinates<3>, NodalVelocities<3>),
        applied_loads: &impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
        masses: &NodalLumpedMasses,
        conditions: &impl ConstrainAt<Solid<3>>,
    ) -> Result<Solution, IntegrationError>
    where
        Self: TimeScaleElements<3>;
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
    fn integrate(
        &self,
        execution: Execution,
        integrator: &impl ExplicitDynamics<
            NodalCoordinates<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        time: &[Quantity<Time>],
        (mut initial_coordinates, initial_velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        applied_loads: &impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
        masses: &impl MassMatrix<3>,
        conditions: &impl ConstrainAt<Solid<3>>,
    ) -> Result<Solution, IntegrationError> {
        let initial_time = *time
            .first()
            .ok_or(IntegrationError::LengthTimeLessThanTwo)?;
        let (masses, initial_velocities, motion) = Motion::held(
            masses,
            conditions,
            initial_time,
            &mut initial_coordinates,
            initial_velocities,
            Stepping::Explicit,
        )?;
        let mut run = execution.start(self)?;
        let (times, mut coordinates, velocities, accelerations) = integrator.integrate(
            |time, coordinates: &NodalCoordinates<3>, _: &NodalVelocities<3>| {
                let mut coordinates = coordinates.clone();
                motion.apply(time, &mut coordinates);
                self.nodal_accelerations_in(&mut run, &coordinates, time, applied_loads, &masses)
                    .map_err(|error| error.to_string())
            },
            time,
            initial_coordinates,
            initial_velocities,
        )?;
        motion.record(&times, &mut coordinates);
        Ok((times, coordinates, velocities, accelerations))
    }
    fn integrate_bounded(
        &self,
        execution: Execution,
        integrator: &impl ExplicitDynamics<
            NodalCoordinates<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        safety: Scalar,
        interval: usize,
        time: &[Quantity<Time>],
        (mut initial_coordinates, initial_velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        applied_loads: &impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
        masses: &NodalLumpedMasses,
        conditions: &impl ConstrainAt<Solid<3>>,
    ) -> Result<Solution, IntegrationError>
    where
        Self: TimeScaleElements<3>,
    {
        if interval == 0 {
            return Err(IntegrationError::Intermediate(
                "The interval between estimates of the time scale must be at least one."
                    .to_string(),
            ));
        }
        let initial_time = *time
            .first()
            .ok_or(IntegrationError::LengthTimeLessThanTwo)?;
        let (masses, initial_velocities, motion) = Motion::held(
            masses,
            conditions,
            initial_time,
            &mut initial_coordinates,
            initial_velocities,
            Stepping::Explicit,
        )?;
        let mut run = execution.start(self)?;
        let mut evaluations = 0;
        let mut time_scale = Time::seconds(Scalar::INFINITY);
        let (times, mut coordinates, velocities, accelerations) = integrator.integrate_bounded(
            |time, coordinates: &NodalCoordinates<3>, _: &NodalVelocities<3>| {
                let mut coordinates = coordinates.clone();
                motion.apply(time, &mut coordinates);
                self.nodal_accelerations_in(&mut run, &coordinates, time, applied_loads, &masses)
                    .map_err(|error| error.to_string())
            },
            |time, coordinates: &NodalCoordinates<3>, _: &NodalVelocities<3>| {
                if evaluations % interval == 0 {
                    let mut coordinates = coordinates.clone();
                    motion.apply(time, &mut coordinates);
                    time_scale = self
                        .fastest_time_scale(&self.coordinates, &coordinates)
                        .map_err(|error| error.to_string())?;
                }
                evaluations += 1;
                Ok(time_scale)
            },
            safety,
            time,
            initial_coordinates,
            initial_velocities,
        )?;
        motion.record(&times, &mut coordinates);
        Ok((times, coordinates, velocities, accelerations))
    }
}
