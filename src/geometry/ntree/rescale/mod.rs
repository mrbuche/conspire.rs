use crate::{
    geometry::{Coordinate, Coordinates, ntree::Orthotree},
    math::{Scalar, Tensor},
};
use std::array::from_fn;

pub struct Rescaling<const D: usize> {
    pub(crate) center: [Scalar; D],
    pub(crate) cell: Scalar,
    pub(crate) half: Scalar,
}

impl<const D: usize> Rescaling<D> {
    /// A tree coordinate is counted in cells rather than measured, so it gives
    /// up its length here and takes one back from the cell it is scaled by.
    pub fn apply(&self, coordinate: &Coordinate<D>) -> Coordinate<D> {
        from_fn(|ax| (coordinate[ax].value() - self.half) * self.cell + self.center[ax]).into()
    }
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V>
    Orthotree<D, L, M, N, T, U, V>
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
