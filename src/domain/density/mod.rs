#[cfg(test)]
mod test;

use crate::{math::Quantity, mechanics::ReferenceCoordinate, units::Density};

#[derive(Clone, Copy, Debug)]
pub struct NoDensity;

pub trait DensityField {
    type Resolved<S>;
    fn density(&self, coordinate: &ReferenceCoordinate) -> Quantity<Density>;
    fn resolve<S>(&self, build: impl FnOnce(&Self) -> S) -> Self::Resolved<S>;
}

impl DensityField for Quantity<Density> {
    type Resolved<S> = Quantity<Density>;
    fn density(&self, _coordinate: &ReferenceCoordinate) -> Quantity<Density> {
        *self
    }
    fn resolve<S>(&self, _build: impl FnOnce(&Self) -> S) -> Quantity<Density> {
        *self
    }
}

impl<F> DensityField for F
where
    F: Fn(&ReferenceCoordinate) -> Quantity<Density>,
{
    type Resolved<S> = S;
    fn density(&self, coordinate: &ReferenceCoordinate) -> Quantity<Density> {
        self(coordinate)
    }
    fn resolve<S>(&self, build: impl FnOnce(&Self) -> S) -> S {
        build(self)
    }
}
