//! What a run keeps on the device, and the two passes that compute the nodal forces with it.

use super::{
    GpuError, GpuItems, GpuOptions, GpuPrecision,
    cuda::{Argument, Buffer, Device, Function, Module},
};
use crate::{
    constitutive::solid::elastic::GpuModel,
    domain::{NodalCoordinates, solid::NodalForcesSolid},
    math::{Quantity, Tensor},
};

const KERNELS: &str = include_str!("kernel.cu");
const THREADS_PER_BLOCK: usize = 128;

/// A floating-point type that a device computes with.
pub(crate) trait Real: Argument + Copy + Default {
    const NAME: &'static str;
    /// Whether the compiler may fuse a multiplication and an addition, which it does not in
    /// `f64`, so that the results stay those of the library as far as the device allows.
    const FUSED: bool;
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl Real for f32 {
    const NAME: &'static str = "float";
    const FUSED: bool = true;
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
    /// Opens the device and puts the blocks on it, for a model of `number_of_nodes` nodes.
    pub(crate) fn new(
        options: GpuOptions,
        blocks: &[GpuItems],
        number_of_nodes: usize,
    ) -> Result<Self, GpuError> {
        match options.precision {
            GpuPrecision::F32 => State::new(options, blocks, number_of_nodes).map(Self::Single),
            GpuPrecision::F64 => State::new(options, blocks, number_of_nodes).map(Self::Double),
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
    reference: Buffer<R>,
    offsets: Buffer<u32>,
    contributions: Buffer<u32>,
    item_forces: Buffer<R>,
    failure: Buffer<u32>,
    forces_kernel: Function,
    gather_kernel: Function,
}

/// The fields are dropped in order, so the buffers go before the kernels and the device.
pub(crate) struct State<R: Real> {
    blocks: Vec<BlockState<R>>,
    coordinates: Buffer<R>,
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
        number_of_nodes: usize,
    ) -> Result<Self, GpuError> {
        let device = Device::new(options.device)?;
        device.bind()?;
        let mut modules = Vec::with_capacity(blocks.len());
        let mut states = Vec::with_capacity(blocks.len());
        for block in blocks {
            let GpuModel::NeoHookean {
                shear_modulus,
                bulk_modulus,
            } = block.model;
            let module = device.module(
                KERNELS,
                &[
                    format!("REAL={}", R::NAME),
                    format!("NODES={}", block.nodes_per_item),
                    format!("POINTS={}", block.points_per_item),
                ],
                R::FUSED,
            )?;
            let items = block.number_of_items();
            let reference = block
                .reference
                .iter()
                .map(|&x| R::from_f64(x))
                .collect::<Vec<_>>();
            states.push(BlockState {
                items: items as u32,
                nodes: (block.offsets.len() - 1) as u32,
                shear_modulus: R::from_f64(shear_modulus),
                bulk_modulus: R::from_f64(bulk_modulus),
                connectivity: device.buffer_of(&block.connectivity)?,
                reference: device.buffer_of(&reference)?,
                offsets: device.buffer_of(&block.offsets)?,
                contributions: device.buffer_of(&block.contributions)?,
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
            forces: device.buffer(3 * number_of_nodes)?,
            host: vec![R::default(); 3 * number_of_nodes],
            nodes: number_of_nodes,
            _modules: modules,
            device,
        })
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
        for block in &self.blocks {
            block.failure.fill(u32::MAX)?;
            self.device.launch(
                &block.forces_kernel,
                block.items as usize,
                THREADS_PER_BLOCK,
                &[
                    &block.items,
                    &block.connectivity,
                    &block.reference,
                    &self.coordinates,
                    &block.item_forces,
                    &block.failure,
                    &block.shear_modulus,
                    &block.bulk_modulus,
                ],
            )?;
            self.device.launch(
                &block.gather_kernel,
                block.nodes as usize,
                THREADS_PER_BLOCK,
                &[
                    &block.nodes,
                    &block.offsets,
                    &block.contributions,
                    &block.item_forces,
                    &self.forces,
                ],
            )?;
        }
        self.device.synchronize()?;
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
