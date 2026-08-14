use crate::{
    geometry::{Coordinate, Coordinates, ntree::Orthotree},
    math::{Quantity, Scalar, Tensor},
    units::Length,
};
use std::array::from_fn;

pub struct Rescaling<const D: usize> {
    pub(crate) center: Coordinate<D>,
    pub(crate) cell: Quantity<Length>,
    pub(crate) half: Scalar,
}

impl<const D: usize> Rescaling<D> {
    pub fn apply(&self, coordinate: &Coordinate<D>) -> Coordinate<D> {
        from_fn(|ax| self.cell * (coordinate[ax].value() - self.half) + self.center[ax]).into()
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
