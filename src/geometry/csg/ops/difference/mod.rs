#[cfg(test)]
mod test;

use super::{D, best_candidate, flip};
use crate::{
    geometry::{
        Coordinate, Direction,
        solid::{Solid, SolidOracle},
    },
    math::Scalar,
};

/// `outer` with `inner` carved out of it.
pub struct Difference<A, B> {
    outer: A,
    inner: B,
}

impl<A, B> Difference<A, B> {
    pub fn new(outer: A, inner: B) -> Self {
        Self { outer, inner }
    }
}

impl<A: Solid, B: Solid> Solid for Difference<A, B> {
    type Oracle = DifferenceOracle<A::Oracle, B::Oracle>;

    fn bounding_box(&self) -> Result<(Coordinate<D>, Coordinate<D>), &'static str> {
        self.outer.bounding_box()
    }

    fn oracle(&self) -> Result<Self::Oracle, &'static str> {
        Ok(DifferenceOracle {
            outer: self.outer.oracle()?,
            inner: self.inner.oracle()?,
        })
    }
}

/// [`SolidOracle`] for a [`Difference`].
pub struct DifferenceOracle<A, B> {
    outer: A,
    inner: B,
}

impl<A: SolidOracle, B: SolidOracle> SolidOracle for DifferenceOracle<A, B> {
    fn signed_distance(&self, query: &Coordinate<D>) -> Scalar {
        self.outer
            .signed_distance(query)
            .min(-self.inner.signed_distance(query))
    }

    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        // Each survival test is at the candidate point, not `query`: the outer
        // wall survives where `inner` has not carved that point away, the
        // cavity wall where that point lies within `outer`.
        best_candidate(
            query,
            [
                self.outer.project(query).map(|(p, n)| {
                    let penalty = self.inner.signed_distance(&p).max(0.0);
                    (p, n, penalty)
                }),
                // Cavity wall: inner's surface within `outer`, normal flipped so
                // it still points out of the solid.
                self.inner.project(query).map(|(p, n)| {
                    let penalty = (-self.outer.signed_distance(&p)).max(0.0);
                    (p, flip(n), penalty)
                }),
            ]
            .into_iter()
            .flatten(),
        )
    }
}
