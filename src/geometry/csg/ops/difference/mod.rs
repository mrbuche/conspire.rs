#[cfg(test)]
mod test;

use super::nearest_surviving;
use crate::{
    geometry::{Coordinate, Direction, solid::SolidOracle},
    math::Quantity,
    units::Length,
};

/// The difference of two solids.
pub struct Difference<A, B> {
    outer: A,
    inner: B,
}

impl<A, B> Difference<A, B> {
    pub fn new(outer: A, inner: B) -> Self {
        Self { outer, inner }
    }
}

impl<A: SolidOracle, B: SolidOracle> SolidOracle for Difference<A, B> {
    fn signed_distance(&self, point: &Coordinate<3>) -> Quantity<Length> {
        self.outer
            .signed_distance(point)
            .max(-self.inner.signed_distance(point))
    }
    fn project(&self, point: &Coordinate<3>) -> Option<(Coordinate<3>, Direction<3>)> {
        let zero = Quantity::default();
        nearest_surviving(
            point,
            [
                self.outer.project(point).map(|(closest, normal)| {
                    let survives = self.inner.signed_distance(&closest) >= zero;
                    (closest, normal, survives)
                }),
                self.inner.project(point).map(|(closest, normal)| {
                    let survives = self.outer.signed_distance(&closest) <= zero;
                    (closest, -normal, survives)
                }),
            ],
        )
    }
}
