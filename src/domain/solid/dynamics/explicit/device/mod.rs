//! A run of explicit dynamics that stays on a device.

use super::super::motion::Motion;
use crate::{
    domain::{
        Model, NodalCoordinates, NodalVelocities, Stepping,
        boundary::{ConstrainAt, Solid},
        gpu::{GpuOptions, Resident},
        solid::{
            elastic::ParallelElasticElements,
            mass::{Lumped, LumpedMassElements, MassChoice},
        },
    },
    math::{Quantity, Tensor, integrate::IntegrationError},
    units::Time,
};
use std::time::Duration;

/// The components of the nodes, as `3 * node + component`.
fn flatten<T, U>(nodes: &T) -> Vec<f64>
where
    T: Tensor,
    T::Item: Tensor<Item = Quantity<U>>,
{
    nodes
        .iter()
        .flat_map(|node| node.iter().map(|component| component.value()))
        .collect()
}

/// A run of explicit dynamics whose coordinates, velocities and forces stay on a device.
///
/// This is the way to time the device alone. [`integrate`](super::ElasticExplicitDynamics::integrate)
/// keeps the history of the run on the calling thread and copies the state between the device and
/// the thread at every step, which costs more than the device saves. Here nothing is copied
/// between the steps, and nothing is kept: only the state at the end comes back.
pub trait DeviceExplicitDynamics {
    /// Takes `steps` steps of velocity Verlet of size `dt` on a device, with lumped masses.
    ///
    /// Degrees of freedom may be held where they are, and none may be moved, as there are no
    /// loads. The coordinates and velocities at the end are returned.
    fn integrate_on_device(
        &self,
        options: GpuOptions,
        dt: Quantity<Time>,
        steps: usize,
        initial_state: (NodalCoordinates<3>, NodalVelocities<3>),
        conditions: &impl ConstrainAt<Solid<3>>,
    ) -> Result<(NodalCoordinates<3>, NodalVelocities<3>), IntegrationError> {
        self.integrate_on_device_timed(options, dt, steps, initial_state, conditions)
            .map(|(state, _)| state)
    }
    /// Takes the steps of [`integrate_on_device`](Self::integrate_on_device), and also returns
    /// how long they took on the device: from when the device has the state and the kernels
    /// ready, which the setup of the run does, to when the last step is done. The setup of
    /// a run is large next to a step, and it varies, so the cost of a step is better read
    /// from this than from the time of the call.
    fn integrate_on_device_timed(
        &self,
        options: GpuOptions,
        dt: Quantity<Time>,
        steps: usize,
        initial_state: (NodalCoordinates<3>, NodalVelocities<3>),
        conditions: &impl ConstrainAt<Solid<3>>,
    ) -> Result<((NodalCoordinates<3>, NodalVelocities<3>), Duration), IntegrationError>;
}

impl<B> DeviceExplicitDynamics for Model<B, 3>
where
    B: ParallelElasticElements<3> + LumpedMassElements<3>,
{
    fn integrate_on_device_timed(
        &self,
        options: GpuOptions,
        dt: Quantity<Time>,
        steps: usize,
        (mut coordinates, velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        conditions: &impl ConstrainAt<Solid<3>>,
    ) -> Result<((NodalCoordinates<3>, NodalVelocities<3>), Duration), IntegrationError> {
        let masses = Lumped.masses(self);
        let (inverse, velocities, motion) = Motion::held(
            &masses,
            conditions,
            Quantity::new(0.0),
            &mut coordinates,
            velocities,
            Stepping::Explicit,
        )?;
        if motion.moves() {
            return Err(IntegrationError::Intermediate(
                "Degrees of freedom that are moved are not supported on a device yet.".to_string(),
            ));
        }
        let error =
            |error: crate::domain::gpu::GpuError| IntegrationError::Intermediate(error.to_string());
        let mut blocks = Vec::new();
        self.gpu_blocks(&mut blocks).map_err(error)?;
        let mut resident = Resident::new(options, &blocks, &self.coordinates).map_err(error)?;
        let (coordinates, velocities, stepping) = resident
            .integrate(
                &inverse.per_degree_of_freedom::<3>(),
                &flatten(&coordinates),
                &flatten(&velocities),
                dt.value(),
                steps,
            )
            .map_err(error)?;
        let unflatten = |values: Vec<f64>| -> Vec<[f64; 3]> { values.as_chunks::<3>().0.to_vec() };
        Ok((
            (unflatten(coordinates).into(), unflatten(velocities).into()),
            stepping,
        ))
    }
}
