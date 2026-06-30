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
    pub fn apply(&self, coordinate: &Coordinate<D>) -> Coordinate<D> {
        from_fn(|ax| (coordinate[ax] - self.half) * self.cell + self.center[ax]).into()
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
