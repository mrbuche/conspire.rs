#[cfg(test)]
mod test;

use super::nearest_surviving;
use crate::{
    geometry::{Coordinate, Direction, solid::SolidOracle},
    math::Quantity,
    units::Length,
};

/// The union of a collection of same-typed solids.
pub struct UnionAll<S>(Vec<S>);

impl<S> UnionAll<S> {
    pub fn new(solids: Vec<S>) -> Self {
        Self(solids)
    }
}

impl<S: SolidOracle> SolidOracle for UnionAll<S> {
    fn signed_distance(&self, point: &Coordinate<3>) -> Quantity<Length> {
        self.0
            .iter()
            .map(|solid| solid.signed_distance(point))
            .fold(Quantity::new(f64::INFINITY), Quantity::min)
    }
    fn project(&self, point: &Coordinate<3>) -> Option<(Coordinate<3>, Direction<3>)> {
        let zero = Quantity::default();
        nearest_surviving(
            point,
            self.0.iter().enumerate().map(|(index, solid)| {
                solid.project(point).map(|(closest, normal)| {
                    let survives = self.0.iter().enumerate().all(|(other, solid)| {
                        other == index || solid.signed_distance(&closest) >= zero
                    });
                    (closest, normal, survives)
                })
            }),
        )
    }
}
