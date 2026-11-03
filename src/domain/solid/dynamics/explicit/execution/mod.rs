//! Where the nodal forces of a run of explicit dynamics are computed.

#[cfg(all(test, feature = "cbm"))]
mod test;

use super::ElasticExplicitDynamics;
#[cfg(feature = "gpu")]
use crate::domain::gpu::{GpuError, GpuOptions, Resident};
use crate::{
    domain::{
        ElementModelError, Model, NodalAccelerations, NodalCoordinates,
        boundary::LoadAt,
        solid::{
            NodalForcesSolid,
            elastic::{ParallelElasticElements, Workers},
            mass::InverseMass,
        },
    },
    math::{Quantity, Tensor, integrate::IntegrationError},
    units::Time,
};
use std::thread::available_parallelism;

/// How the nodal forces of a run of explicit dynamics are computed.
///
/// The choice never changes the results of [`Serial`](Self::Serial) and
/// [`Threads`](Self::Threads), which are identical, and only the run is affected: the threads and
/// their storage are created when the run starts and dropped when it ends. A device agrees with
/// them to the precision that it computes with.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Execution {
    /// On the calling thread, with nothing created for the run.
    #[default]
    Serial,
    /// On the given number of threads, which live for the run. Any number, including one,
    /// takes the parallel pathway, which [`Serial`](Self::Serial) avoids, and zero is as many
    /// threads as the machine offers.
    Threads(usize),
    /// On a CUDA device, where the blocks and the nodal forces live for the run. Every block
    /// must be one whose constitutive model and elements the device has kernels for, and the
    /// run fails when it starts if one is not, or if there is no device. The loads and the
    /// masses are still those of the calling thread.
    #[cfg(feature = "gpu")]
    Gpu(GpuOptions),
}

/// The threads and storage of one run.
pub(super) enum Run<S> {
    Serial,
    Parallel(Workers, S),
    #[cfg(feature = "gpu")]
    Gpu(Resident),
}

impl Execution {
    pub(super) fn start<B>(self, model: &Model<B, 3>) -> Result<Run<B::Scratch>, IntegrationError>
    where
        B: ParallelElasticElements<3>,
    {
        let threads = match self {
            Self::Serial => return Ok(Run::Serial),
            Self::Threads(0) => available_parallelism().map_or(1, usize::from),
            Self::Threads(threads) => threads,
            #[cfg(feature = "gpu")]
            Self::Gpu(options) => {
                let mut blocks = Vec::new();
                return model
                    .gpu_blocks(&mut blocks)
                    .and_then(|()| Resident::new(options, &blocks, model.coordinates.len()))
                    .map(Run::Gpu)
                    .map_err(|error| IntegrationError::Intermediate(error.to_string()));
            }
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
    pub(super) fn nodal_accelerations_in(
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
            #[cfg(feature = "gpu")]
            Run::Gpu(resident) => {
                let mut unbalanced_forces = NodalForcesSolid::zero(nodal_coordinates.len());
                resident
                    .nodal_forces_into(nodal_coordinates, &mut unbalanced_forces)
                    .map_err(|error| match error {
                        GpuError::Item { block, item } => self
                            .gpu_item_error(block, item, nodal_coordinates)
                            .unwrap_or_else(|| ElementModelError::upstream(error, "the device")),
                        error => ElementModelError::upstream(error, "the device"),
                    })?;
                applied_loads
                    .subtract_at(time, nodal_coordinates, &mut unbalanced_forces)
                    .map_err(|error| ElementModelError::upstream(error, "the applied loads"))?;
                Ok(masses.nodal_accelerations(&unbalanced_forces))
            }
        }
    }
}
