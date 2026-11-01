use super::{Solution, motion::Motion};
use crate::{
    domain::{
        ElementModelError, Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory, Stepping,
        boundary::{ConstrainAt, LoadAt, Solid},
        solid::{
            NodalForcesSolid,
            elastic::{ElasticElements, ParallelElasticElements, Workers},
            mass::{
                ConsistentMassElements, InverseMass, LumpedMassElements, MassMatrix,
                NodalLumpedMasses, NodalMasses,
            },
        },
    },
    math::{
        Quantity, Tensor,
        integrate::{ExplicitDynamics, IntegrationError},
    },
    units::Time,
};
use std::thread::available_parallelism;

mod stability;

pub use stability::{Checked, Stability, Unchecked};

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

/// Which masses a run of explicit dynamics uses, taken from the blocks of the model.
///
/// The model makes the masses from the same densities that the estimate of the stability limit
/// uses, so the two cannot disagree, and none are passed in.
pub trait MassChoice<M> {
    /// The masses of the choice.
    type Mass: MassMatrix<3>;
    /// The masses of the model.
    fn masses(self, model: &M) -> Self::Mass;
}

/// The lumped masses of the model, which need a division to give the accelerations.
#[derive(Clone, Copy, Debug, Default)]
pub struct Lumped;

/// The consistent masses of the model, which need a solve to give the accelerations.
#[derive(Clone, Copy, Debug, Default)]
pub struct Consistent;

impl<B> MassChoice<Model<B, 3>> for Lumped
where
    B: LumpedMassElements<3>,
{
    type Mass = NodalLumpedMasses;
    fn masses(self, model: &Model<B, 3>) -> NodalLumpedMasses {
        model.nodal_lumped_masses()
    }
}

impl<B> MassChoice<Model<B, 3>> for Consistent
where
    B: ConsistentMassElements,
{
    type Mass = NodalMasses;
    fn masses(self, model: &Model<B, 3>) -> NodalMasses {
        model.nodal_masses()
    }
}

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
