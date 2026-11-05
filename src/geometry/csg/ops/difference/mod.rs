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
    fn project(&self, point: &Coordinate<3>) -> Option<Projection> {
        let zero = Quantity::default();
        nearest_surviving(
            point,
            [
                self.outer.project(point).map(|projection| {
                    let survives = self.inner.signed_distance(&projection.point) >= zero;
                    (projection, survives)
                }),
                self.inner.project(point).map(|projection| {
                    let survives = self.outer.signed_distance(&projection.point) <= zero;
                    let normal = -projection.normal.clone();
                    (
                        Projection {
                            normal,
                            ..projection
                        },
                        survives,
                    )
                }),
            ],
        )
    }
}
