//! What a run keeps on the device, and the two passes that compute the nodal forces with it.

use super::{
    GpuError, GpuGradients, GpuItems, GpuOptions, GpuPrecision,
    cuda::{Argument, Buffer, Device, Function, Module},
};
use crate::{
    constitutive::solid::elastic::GpuModel,
    domain::{NodalCoordinates, NodalReferenceCoordinates, solid::NodalForcesSolid},
    math::{Quantity, Tensor},
};

const KERNELS: &str = include_str!("kernel.cu");
const DYNAMICS: &str = include_str!("dynamics.cu");
const THREADS_PER_BLOCK: usize = 128;
/// How many steps of a run on the device are taken between looks for an inverted element.
const CHECK_EVERY: usize = 1024;

/// A floating-point type that a device computes with.
pub(crate) trait Real: Argument + Copy + Default {
    const NAME: &'static str;
    /// Whether the compiler may fuse a multiplication and an addition, which it does not in
    /// `f64`, so that the results stay those of the library as far as the device allows.
    const FUSED: bool;
    /// Whether this is double precision, for which some operations are done differently.
    const DOUBLE: bool;
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl Real for f32 {
    const NAME: &'static str = "float";
    const FUSED: bool = true;
    const DOUBLE: bool = false;
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    fn to_f64(self) -> f64 {
        self.into()
    }
}

impl Real for f64 {
    const NAME: &'static str = "double";
    const FUSED: bool = false;
    const DOUBLE: bool = true;
    fn from_f64(value: f64) -> Self {
        value
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// The device, with the buffers and kernels of the blocks of a model, for one run.
pub(crate) enum Resident {
    Single(State<f32>),
    Double(State<f64>),
}

impl Resident {
    /// Opens the device and puts the blocks on it, for a model with the given reference
    /// coordinates of its nodes.
    pub(crate) fn new(
        options: GpuOptions,
        blocks: &[GpuItems],
        reference_coordinates: &NodalReferenceCoordinates<3>,
    ) -> Result<Self, GpuError> {
        let reference_coordinates = reference_coordinates
            .iter()
            .flat_map(|coordinate| coordinate.iter().map(|component| component.value()))
            .collect::<Vec<f64>>();
        match options.precision {
            GpuPrecision::F32 => {
                State::new(options, blocks, &reference_coordinates).map(Self::Single)
            }
            GpuPrecision::F64 => {
                State::new(options, blocks, &reference_coordinates).map(Self::Double)
            }
        }
    }
    /// The name of the device.
    #[cfg(test)]
    pub(crate) fn device_name(&self) -> &str {
        match self {
            Self::Single(state) => state.device.name(),
            Self::Double(state) => state.device.name(),
        }
    }
    /// Takes `steps` steps of velocity Verlet of size `dt` entirely on the device, from the
    /// coordinates and velocities as `3 * node + component`, with the masses of each of those
    /// degrees of freedom, which are infinite for those that are held. Only the coordinates and
    /// velocities at the end come back, so nothing is copied between steps.
    pub(crate) fn integrate(
        &mut self,
        masses: &[f64],
        coordinates: &[f64],
        velocities: &[f64],
        dt: f64,
        steps: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
        match self {
            Self::Single(state) => state.integrate(masses, coordinates, velocities, dt, steps),
            Self::Double(state) => state.integrate(masses, coordinates, velocities, dt, steps),
        }
    }
    /// Adds the nodal forces to `nodal_forces`, in the order of the elements for each node.
    ///
    /// An element whose Jacobian is not positive is an error, and the first such element is the
    /// one that is reported.
    pub(crate) fn nodal_forces_into(
        &mut self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), GpuError> {
        match self {
            Self::Single(state) => state.nodal_forces_into(nodal_coordinates, nodal_forces),
            Self::Double(state) => state.nodal_forces_into(nodal_coordinates, nodal_forces),
        }
    }
}

struct BlockState<R> {
    items: u32,
    nodes: u32,
    shear_modulus: R,
    bulk_modulus: R,
    connectivity: Buffer<u32>,
    geometry: Geometry<R>,
    offsets: Buffer<u32>,
    contributions: Buffer<u32>,
    item_forces: Buffer<R>,
    failure: Buffer<u32>,
    forces_kernel: Function,
    gather_kernel: Function,
}

/// What the kernel of a block is given to get the gradient vectors from.
enum Geometry<R> {
    /// The gradient vectors and weights of each element.
    Stored { reference: Buffer<R> },
    /// The derivatives of the shape functions and the weights of the parametric element, which
    /// are shared by the elements.
    OnTheFly {
        derivatives: Buffer<R>,
        weights: Buffer<R>,
    },
}

/// The fields are dropped in order, so the buffers go before the kernels and the device.
pub(crate) struct State<R: Real> {
    blocks: Vec<BlockState<R>>,
    coordinates: Buffer<R>,
    reference_coordinates: Option<Buffer<R>>,
    forces: Buffer<R>,
    host: Vec<R>,
    nodes: usize,
    _modules: Vec<Module>,
    device: Device,
}

impl<R: Real> State<R> {
    fn new(
        options: GpuOptions,
        blocks: &[GpuItems],
        reference_coordinates: &[f64],
    ) -> Result<Self, GpuError> {
        let number_of_nodes = reference_coordinates.len() / 3;
        let on_the_fly = options.gradients == GpuGradients::OnTheFly;
        let device = Device::new(options.device)?;
        device.bind()?;
        let to_device = |values: &[f64]| values.iter().map(|&x| R::from_f64(x)).collect::<Vec<_>>();
        let mut modules = Vec::with_capacity(blocks.len());
        let mut states = Vec::with_capacity(blocks.len());
        for block in blocks {
            let GpuModel::NeoHookean {
                shear_modulus,
                bulk_modulus,
            } = block.model;
            let mut definitions = vec![
                format!("REAL={}", R::NAME),
                format!("NODES={}", block.nodes_per_item),
                format!("POINTS={}", block.points_per_item),
            ];
            if on_the_fly {
                definitions.push("ON_THE_FLY".to_string());
            }
            if R::DOUBLE {
                definitions.push("DOUBLE_PRECISION".to_string());
            }
            let module = device.module(KERNELS, &definitions, R::FUSED)?;
            let items = block.number_of_items();
            let layout = block.device_layout();
            let geometry = if on_the_fly {
                let shape = block.shape.as_ref().ok_or_else(|| {
                    GpuError::Unsupported(
                        "the elements cannot make their gradient vectors on the fly".to_string(),
                    )
                })?;
                Geometry::OnTheFly {
                    derivatives: device.buffer_of(&to_device(&shape.derivatives))?,
                    weights: device.buffer_of(&to_device(&shape.weights))?,
                }
            } else {
                Geometry::Stored {
                    reference: device.buffer_of(&to_device(&layout.reference))?,
                }
            };
            states.push(BlockState {
                items: items as u32,
                nodes: (block.offsets.len() - 1) as u32,
                shear_modulus: R::from_f64(shear_modulus),
                bulk_modulus: R::from_f64(bulk_modulus),
                connectivity: device.buffer_of(&layout.connectivity)?,
                geometry,
                offsets: device.buffer_of(&block.offsets)?,
                contributions: device.buffer_of(&layout.contributions)?,
                item_forces: device.buffer(items * block.nodes_per_item * 3)?,
                failure: device.buffer::<u32>(1)?,
                forces_kernel: module.function("item_forces")?,
                gather_kernel: module.function("gather")?,
            });
            modules.push(module);
        }
        Ok(Self {
            blocks: states,
            coordinates: device.buffer(3 * number_of_nodes)?,
            reference_coordinates: if on_the_fly {
                Some(device.buffer_of(&to_device(reference_coordinates))?)
            } else {
                None
            },
            forces: device.buffer(3 * number_of_nodes)?,
            host: vec![R::default(); 3 * number_of_nodes],
            nodes: number_of_nodes,
            _modules: modules,
            device,
        })
    }
    /// Marks that no element has failed.
    fn reset_failures(&self) -> Result<(), GpuError> {
        self.blocks
            .iter()
            .try_for_each(|block| block.failure.fill(u32::MAX))
    }
    /// Launches the passes that add the nodal forces of the coordinates on the device to the
    /// forces on the device, without waiting for them.
    fn launch_forces(&self) -> Result<(), GpuError> {
        for block in &self.blocks {
            match (&block.geometry, &self.reference_coordinates) {
                (Geometry::Stored { reference }, _) => self.device.launch(
                    &block.forces_kernel,
                    block.items as usize,
                    THREADS_PER_BLOCK,
                    &[
                        &block.items,
                        &block.connectivity,
                        reference,
                        &self.coordinates,
                        &block.item_forces,
                        &block.failure,
                        &block.shear_modulus,
                        &block.bulk_modulus,
                    ],
                ),
                (
                    Geometry::OnTheFly {
                        derivatives,
                        weights,
                    },
                    Some(reference_coordinates),
                ) => self.device.launch(
                    &block.forces_kernel,
                    block.items as usize,
                    THREADS_PER_BLOCK,
                    &[
                        &block.items,
                        &block.connectivity,
                        reference_coordinates,
                        derivatives,
                        weights,
                        &self.coordinates,
                        &block.item_forces,
                        &block.failure,
                        &block.shear_modulus,
                        &block.bulk_modulus,
                    ],
                ),
                (Geometry::OnTheFly { .. }, None) => Err(GpuError::Unsupported(
                    "the reference coordinates are not on the device".to_string(),
                )),
            }?;
            self.device.launch(
                &block.gather_kernel,
                block.nodes as usize,
                THREADS_PER_BLOCK,
                &[
                    &block.nodes,
                    &block.items,
                    &block.offsets,
                    &block.contributions,
                    &block.item_forces,
                    &self.forces,
                ],
            )?;
        }
        Ok(())
    }
    /// Fails with the first element that was found to be inverted since the failures were reset.
    /// Reading the marks waits for what was launched.
    fn check_failures(&self) -> Result<(), GpuError> {
        for (index, block) in self.blocks.iter().enumerate() {
            let mut failure = [u32::MAX];
            block.failure.download(&mut failure)?;
            if failure[0] != u32::MAX {
                return Err(GpuError::Item {
                    block: index,
                    item: failure[0] as usize,
                });
            }
        }
        Ok(())
    }
    /// Takes `steps` steps of velocity Verlet on the device, and returns the coordinates and
    /// velocities at the end, which are all that comes back.
    ///
    /// The masses are those of each degree of freedom, infinite for those held. Nothing is
    /// copied between the device and the host during the run, and the elements are checked
    /// every `CHECK_EVERY` steps.
    fn integrate(
        &mut self,
        masses: &[f64],
        coordinates: &[f64],
        velocities: &[f64],
        dt: f64,
        steps: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
        let count = 3 * self.nodes;
        if [masses.len(), coordinates.len(), velocities.len()] != [count; 3] {
            return Err(GpuError::Unsupported(format!(
                "the masses, coordinates and velocities are not each {count} values"
            )));
        }
        self.device.bind()?;
        let to_device = |values: &[f64]| values.iter().map(|&x| R::from_f64(x)).collect::<Vec<_>>();
        let masses = self.device.buffer_of(&to_device(masses))?;
        let velocities = self.device.buffer_of(&to_device(velocities))?;
        let accelerations = self.device.buffer::<R>(count)?;
        self.coordinates.upload(&to_device(coordinates))?;
        self.forces.fill(0)?;
        self.reset_failures()?;
        let module = self
            .device
            .module(DYNAMICS, &[format!("REAL={}", R::NAME)], R::FUSED)?;
        let kick_drift = module.function("kick_drift")?;
        let accelerate = module.function("accelerate")?;
        let (count, dt) = (count as u32, R::from_f64(dt));
        let (first, kick) = (0_u32, 1_u32);
        // The accelerations at the start, which do not change the velocities.
        self.launch_forces()?;
        self.device.launch(
            &accelerate,
            count as usize,
            THREADS_PER_BLOCK,
            &[
                &count,
                &self.forces,
                &masses,
                &accelerations,
                &velocities,
                &dt,
                &first,
            ],
        )?;
        for step in 0..steps {
            self.device.launch(
                &kick_drift,
                count as usize,
                THREADS_PER_BLOCK,
                &[&count, &self.coordinates, &velocities, &accelerations, &dt],
            )?;
            self.launch_forces()?;
            self.device.launch(
                &accelerate,
                count as usize,
                THREADS_PER_BLOCK,
                &[
                    &count,
                    &self.forces,
                    &masses,
                    &accelerations,
                    &velocities,
                    &dt,
                    &kick,
                ],
            )?;
            if (step + 1) % CHECK_EVERY == 0 {
                self.check_failures()?;
            }
        }
        self.device.synchronize()?;
        self.check_failures()?;
        let mut final_coordinates = vec![R::default(); count as usize];
        let mut final_velocities = vec![R::default(); count as usize];
        self.coordinates.download(&mut final_coordinates)?;
        velocities.download(&mut final_velocities)?;
        let to_host = |values: Vec<R>| values.into_iter().map(Real::to_f64).collect();
        Ok((to_host(final_coordinates), to_host(final_velocities)))
    }
    fn nodal_forces_into(
        &mut self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), GpuError> {
        if nodal_coordinates.len() != self.nodes {
            return Err(GpuError::Unsupported(format!(
                "{} nodal coordinates were given to a run of {} nodes",
                nodal_coordinates.len(),
                self.nodes
            )));
        }
        self.device.bind()?;
        nodal_coordinates
            .iter()
            .zip(self.host.as_chunks_mut::<3>().0)
            .for_each(|(coordinate, host)| {
                host.iter_mut()
                    .zip(coordinate.iter())
                    .for_each(|(host, component)| *host = R::from_f64(component.value()))
            });
        self.coordinates.upload(&self.host)?;
        self.forces.fill(0)?;
        self.reset_failures()?;
        self.launch_forces()?;
        self.device.synchronize()?;
        self.check_failures()?;
        self.forces.download(&mut self.host)?;
        nodal_forces
            .iter_mut()
            .zip(self.host.as_chunks::<3>().0)
            .for_each(|(nodal_force, host)| {
                nodal_force
                    .iter_mut()
                    .zip(host)
                    .for_each(|(component, host)| *component += Quantity::new(host.to_f64()))
            });
        Ok(())
    }
}

impl<R: Real> Drop for State<R> {
    fn drop(&mut self) {
        // The buffers are freed in the context of the device, on whichever thread drops this.
        let _ = self.device.bind();
    }
}
