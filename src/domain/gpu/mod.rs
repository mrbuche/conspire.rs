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
use core::fmt::{self, Display, Formatter};

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

/// Where the device gets the gradient vectors of the shape functions of the elements.
///
/// Reading them costs memory traffic, and making them costs arithmetic, so which is faster
/// depends on the device and the precision: a device that is limited by its memory does better to
/// make them, and one that is limited by its arithmetic, as many are in `f64`, to read them.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GpuGradients {
    /// Each element keeps its own, which are read at every step.
    #[default]
    Stored,
    /// They are made at every step from the reference coordinates of the nodes of the element,
    /// as an element makes them when it is made. Only elements of the three dimensions whose
    /// shape functions are those of their nodes can do this.
    OnTheFly,
}

/// How a device is used for a run.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuOptions {
    /// The ordinal of the device, as CUDA numbers them.
    pub device: usize,
    /// The precision that the device computes with.
    pub precision: GpuPrecision,
    /// Where the device gets the gradient vectors.
    pub gradients: GpuGradients,
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

impl core::error::Error for GpuError {}

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
    /// These arrays are by element, and [`device_layout`](Self::device_layout) puts the
    /// element last for the kernels.
    pub contributions: Vec<u32>,
    /// What makes the gradient vectors from the reference coordinates, if the elements can.
    pub shape: Option<ShapeTables>,
}

/// The shape functions of the elements of a block, at the integration points of its parametric
/// element, from which gradient vectors are made on the fly.
pub struct ShapeTables {
    /// The derivatives of the shape functions with respect to the parametric coordinates, as
    /// point, then node, then component.
    pub derivatives: Vec<f64>,
    /// The weights of the integration points of the parametric element.
    pub weights: Vec<f64>,
}

impl GpuItems {
    pub fn number_of_items(&self) -> usize {
        self.connectivity.len() / self.nodes_per_item
    }
    /// The arrays as the kernels read them, with the element last, so that the threads of a
    /// warp, which are consecutive elements, read consecutive addresses.
    pub(crate) fn device_layout(&self) -> DeviceLayout {
        let items = self.number_of_items();
        let reference_size = self.reference.len() / items.max(1);
        DeviceLayout {
            connectivity: transpose(&self.connectivity, self.nodes_per_item, items),
            reference: transpose(&self.reference, reference_size, items),
            // The position of the node in the element, then the element, instead of the
            // element and then the position.
            contributions: self
                .contributions
                .iter()
                .map(|&contribution| {
                    let (item, local) = (
                        contribution as usize / self.nodes_per_item,
                        contribution as usize % self.nodes_per_item,
                    );
                    (local * items + item) as u32
                })
                .collect(),
        }
    }
}

/// The arrays of a block as the kernels read them.
pub(crate) struct DeviceLayout {
    /// The node of each element, as `position * items + item`.
    pub connectivity: Vec<u32>,
    /// The reference values of each element, as `value * items + item`.
    pub reference: Vec<f64>,
    /// The contributions to the nodes, as `position * items + item`.
    pub contributions: Vec<u32>,
}

/// `values`, which are `width` for each of `items`, with the item last instead of first.
fn transpose<T: Copy + Default>(values: &[T], width: usize, items: usize) -> Vec<T> {
    let mut transposed = vec![T::default(); values.len()];
    values
        .chunks_exact(width.max(1))
        .enumerate()
        .for_each(|(item, row)| {
            row.iter()
                .enumerate()
                .for_each(|(value, &x)| transposed[value * items + item] = x)
        });
    transposed
}
