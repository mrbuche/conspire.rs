//! Explicit dynamics on a CUDA device.
//!
//! The nodal forces of a run are computed on the device in two passes, as on the threads: the
//! forces of the elements, and then the sum of those that each node has, in the order of the
//! elements. What does not change between steps lives on the device for the run.

mod cuda;
mod resident;
#[cfg(test)]
mod test;

pub(crate) use resident::Resident;

use crate::constitutive::solid::elastic::GpuModel;
use std::fmt::{self, Display, Formatter};

/// The floating-point type that a device computes with.
///
/// Many devices compute `f64` at a small fraction of the rate of `f32`, and the library
/// computes with `f64`, so that a device in `F32` agrees with it only to `f32`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GpuPrecision {
    /// Double precision.
    #[default]
    F64,
    /// Single precision.
    F32,
}

/// How a device is used for a run.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuOptions {
    /// The ordinal of the device, as CUDA numbers them.
    pub device: usize,
    /// The precision that the device computes with.
    pub precision: GpuPrecision,
}

/// What can go wrong on a device.
#[derive(Debug)]
pub enum GpuError {
    /// There is no CUDA, or no such device.
    Unavailable(String),
    /// The model or its elements have no kernels.
    Unsupported(String),
    /// The driver or the compiler reported an error.
    Cuda(String),
    /// An element of a block has a Jacobian that is not positive.
    Item {
        /// The block, in the order of the model.
        block: usize,
        /// The element of the block.
        item: usize,
    },
}

impl Display for GpuError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(message) => write!(f, "No CUDA device is available: {message}."),
            Self::Unsupported(message) => write!(f, "The device cannot compute this: {message}."),
            Self::Cuda(message) => write!(f, "CUDA: {message}"),
            Self::Item { block, item } => write!(
                f,
                "Element {item} of block {block} has a Jacobian that is not positive."
            ),
        }
    }
}

impl std::error::Error for GpuError {}

/// A block of elements as a device takes it: plain arrays, in the order of the elements.
pub struct GpuItems {
    /// The constitutive model of the block.
    pub model: GpuModel,
    /// The nodes of each element.
    pub nodes_per_item: usize,
    /// The integration points of each element.
    pub points_per_item: usize,
    /// The nodes of the elements, `nodes_per_item` for each.
    pub connectivity: Vec<u32>,
    /// For each element, the gradient vectors of its nodes at each of its integration points
    /// (point, then node, then component), and then the weights of the points.
    pub reference: Vec<f64>,
    /// The offsets of the contributions to each node, from the index of the elements.
    pub offsets: Vec<u32>,
    /// The contributions to the nodes, as `nodes_per_item * element + position of the node`.
    pub contributions: Vec<u32>,
}

impl GpuItems {
    pub fn number_of_items(&self) -> usize {
        self.connectivity.len() / self.nodes_per_item
    }
}
