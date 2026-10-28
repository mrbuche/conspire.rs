#[cfg(test)]
mod test;

use super::nearest_surviving;
use crate::{
    geometry::{Coordinate, Direction, solid::SolidOracle},
    math::Quantity,
    units::Length,
};

/// The union of two solids.
pub struct Union<A, B> {
    first: A,
    second: B,
}

impl<A, B> Union<A, B> {
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A: SolidOracle, B: SolidOracle> SolidOracle for Union<A, B> {
    fn signed_distance(&self, point: &Coordinate<3>) -> Quantity<Length> {
        self.first
            .signed_distance(point)
            .min(self.second.signed_distance(point))
    }
    fn project(&self, point: &Coordinate<3>) -> Option<(Coordinate<3>, Direction<3>)> {
        let zero = Quantity::default();
        nearest_surviving(
            point,
            [
                self.first.project(point).map(|(closest, normal)| {
                    let survives = self.second.signed_distance(&closest) >= zero;
                    (closest, normal, survives)
                }),
                self.second.project(point).map(|(closest, normal)| {
                    let survives = self.first.signed_distance(&closest) >= zero;
                    (closest, normal, survives)
                }),
            ],
        )
    }
}
