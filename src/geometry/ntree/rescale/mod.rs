use crate::{
    geometry::{Coordinate, Coordinates, ntree::Orthotree},
    math::{Scalar, Tensor},
};
use std::array::from_fn;

/// Affine map from the integer coordinates of an [`Orthotree`](super::Orthotree)
/// back to the real-space coordinates of whatever it was built to discretize.
///
/// A real point maps to the tree as `integer = (point - center) / cell + half`,
/// so the inverse applied here is `point = (integer - half) * cell + center`.
pub struct Rescaling<const D: usize> {
    pub(crate) center: [Scalar; D],
    pub(crate) cell: Scalar,
    pub(crate) half: Scalar,
}

impl<const D: usize> Rescaling<D> {
    /// Maps a coordinate from tree space back to real space.
    pub fn apply(&self, coordinate: &Coordinate<D>) -> Coordinate<D> {
        from_fn(|ax| (coordinate[ax] - self.half) * self.cell + self.center[ax]).into()
    }
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U>
    Orthotree<D, L, M, N, T, U>
{
    pub fn rescale(&self) -> &Rescaling<D> {
        &self.rescale
    }
    pub fn rescale_coordinates(&self, coordinates: &mut Coordinates<D>) {
        coordinates
            .iter_mut()
            .for_each(|coordinate| *coordinate = self.rescale.apply(coordinate));
    }
}
