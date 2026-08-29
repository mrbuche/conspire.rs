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

/// The union of a runtime-sized list of same-typed solids. For a fixed mix of
/// different types use [`Union`](super::Union) instead.
///
/// An empty list is the empty set: nothing is inside it and it has no bounding
/// box, which is only useful as the subtrahend of a [`Difference`](super::Difference).
pub struct UnionAll<S>(Vec<S>);

impl<S> UnionAll<S> {
    pub fn new(solids: Vec<S>) -> Self {
        Self(solids)
    }
}

impl<S: Solid> Solid for UnionAll<S> {
    type Oracle = UnionAllOracle<S::Oracle>;

    fn bounding_box(&self) -> Result<(Coordinate<D>, Coordinate<D>), &'static str> {
        if self.0.is_empty() {
            return Err("an empty union has no bounding box");
        }
        let mut low = [Scalar::INFINITY; D];
        let mut high = [Scalar::NEG_INFINITY; D];
        for solid in &self.0 {
            let (l, h) = solid.bounding_box()?;
            for k in 0..D {
                low[k] = low[k].min(l[k].value());
                high[k] = high[k].max(h[k].value());
            }
        }
        Ok((low.into(), high.into()))
    }

    fn oracle(&self) -> Result<Self::Oracle, &'static str> {
        Ok(UnionAllOracle(
            self.0
                .iter()
                .map(Solid::oracle)
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

/// [`SolidOracle`] for a [`UnionAll`].
pub struct UnionAllOracle<O>(Vec<O>);

impl<O: SolidOracle> SolidOracle for UnionAllOracle<O> {
    fn signed_distance(&self, query: &Coordinate<D>) -> Scalar {
        self.0
            .iter()
            .map(|oracle| oracle.signed_distance(query))
            .fold(Scalar::NEG_INFINITY, Scalar::max)
    }

    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        let signed: Vec<Scalar> = self
            .0
            .iter()
            .map(|oracle| oracle.signed_distance(query))
            .collect();
        best_candidate(
            query,
            self.0.iter().enumerate().filter_map(|(index, oracle)| {
                let (point, normal) = oracle.project(query)?;
                // A patch of this operand's surface survives where no other
                // operand encloses the query.
                let valid = signed
                    .iter()
                    .enumerate()
                    .all(|(other, &distance)| other == index || distance <= 0.0);
                Some((point, normal, valid))
            }),
        )
    }
}
