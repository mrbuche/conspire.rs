#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate,
        solid::{Projection, SolidOracle, nearest_surviving},
    },
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
    fn project(&self, point: &Coordinate<3>) -> Option<Projection> {
        let zero = Quantity::default();
        nearest_surviving(
            point,
            [
                self.first.project(point).map(|projection| {
                    let survives = self.second.signed_distance(&projection.point) >= zero;
                    (projection, survives)
                }),
                self.second.project(point).map(|projection| {
                    let survives = self.first.signed_distance(&projection.point) >= zero;
                    (projection, survives)
                }),
            ],
        )
    }
}
