#[cfg(test)]
mod test;

use super::{D, best_candidate};
use crate::{
    geometry::{
        Coordinate, Direction,
        solid::{Solid, SolidOracle},
    },
    math::Scalar,
};
use std::array::from_fn;

/// The intersection of two solids: inside when inside both.
pub struct Intersection<A, B> {
    a: A,
    b: B,
}

impl<A, B> Intersection<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: Solid, B: Solid> Solid for Intersection<A, B> {
    type Oracle = IntersectionOracle<A::Oracle, B::Oracle>;

    fn bounding_box(&self) -> Result<(Coordinate<D>, Coordinate<D>), &'static str> {
        let (la, ha) = self.a.bounding_box()?;
        let (lb, hb) = self.b.bounding_box()?;
        let low: [Scalar; D] = from_fn(|k| la[k].value().max(lb[k].value()));
        let high: [Scalar; D] = from_fn(|k| ha[k].value().min(hb[k].value()));
        if (0..D).any(|k| low[k] >= high[k]) {
            return Err("solids do not overlap");
        }
        Ok((low.into(), high.into()))
    }

    fn oracle(&self) -> Result<Self::Oracle, &'static str> {
        Ok(IntersectionOracle {
            a: self.a.oracle()?,
            b: self.b.oracle()?,
        })
    }
}

/// [`SolidOracle`] for an [`Intersection`].
pub struct IntersectionOracle<A, B> {
    a: A,
    b: B,
}

impl<A: SolidOracle, B: SolidOracle> SolidOracle for IntersectionOracle<A, B> {
    fn signed_distance(&self, query: &Coordinate<D>) -> Scalar {
        self.a
            .signed_distance(query)
            .min(self.b.signed_distance(query))
    }

    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        // A patch of one operand's surface survives the intersection where the
        // *other* operand also encloses that patch point; the penalty is how
        // far outside the other it is.
        best_candidate(
            query,
            [
                self.a.project(query).map(|(p, n)| {
                    let penalty = (-self.b.signed_distance(&p)).max(0.0);
                    (p, n, penalty)
                }),
                self.b.project(query).map(|(p, n)| {
                    let penalty = (-self.a.signed_distance(&p)).max(0.0);
                    (p, n, penalty)
                }),
            ]
            .into_iter()
            .flatten(),
        )
    }
}
