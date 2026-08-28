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

/// The union of two solids: inside when inside either. Binary and generic, so an
/// operand may itself be a combinator or any other [`Solid`]; nest for a higher
/// arity.
pub struct Union<A, B> {
    a: A,
    b: B,
}

impl<A, B> Union<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: Solid, B: Solid> Solid for Union<A, B> {
    type Oracle = UnionOracle<A::Oracle, B::Oracle>;

    fn bounding_box(&self) -> Result<(Coordinate<D>, Coordinate<D>), &'static str> {
        let (la, ha) = self.a.bounding_box()?;
        let (lb, hb) = self.b.bounding_box()?;
        let low: [Scalar; D] = from_fn(|k| la[k].value().min(lb[k].value()));
        let high: [Scalar; D] = from_fn(|k| ha[k].value().max(hb[k].value()));
        Ok((low.into(), high.into()))
    }

    fn oracle(&self) -> Result<Self::Oracle, &'static str> {
        Ok(UnionOracle {
            a: self.a.oracle()?,
            b: self.b.oracle()?,
        })
    }
}

/// [`SolidOracle`] for a [`Union`].
pub struct UnionOracle<A, B> {
    a: A,
    b: B,
}

impl<A: SolidOracle, B: SolidOracle> SolidOracle for UnionOracle<A, B> {
    fn signed_distance(&self, query: &Coordinate<D>) -> Scalar {
        self.a
            .signed_distance(query)
            .max(self.b.signed_distance(query))
    }

    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        let (sa, sb) = (self.a.signed_distance(query), self.b.signed_distance(query));
        best_candidate(
            query,
            [
                self.a.project(query).map(|(p, n)| (p, n, sb <= 0.0)),
                self.b.project(query).map(|(p, n)| (p, n, sa <= 0.0)),
            ]
            .into_iter()
            .flatten(),
        )
    }
}
