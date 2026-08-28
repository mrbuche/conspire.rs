//! Boolean combinations of solids. Each combinator is itself a
//! [`Solid`](crate::geometry::solid::Solid), so `Difference::new(box, pores)`
//! meshes through the same driver as a primitive.

#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Direction,
        solid::{Solid, SolidOracle},
    },
    math::Scalar,
};
use std::array::from_fn;

const D: usize = 3;

// ---------------------------------------------------------------- Union --------

/// The union of one or more solids: inside when inside any operand.
pub struct Union<S>(Vec<S>);

impl<S> Union<S> {
    /// A union over a non-empty list of solids.
    pub fn new(solids: Vec<S>) -> Result<Self, &'static str> {
        if solids.is_empty() {
            return Err("union needs at least one solid");
        }
        Ok(Self(solids))
    }
}

impl<S: Solid> Solid for Union<S> {
    type Oracle = UnionOracle<S::Oracle>;

    fn bounding_box(&self) -> Result<(Coordinate<D>, Coordinate<D>), &'static str> {
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
        Ok(UnionOracle(
            self.0
                .iter()
                .map(Solid::oracle)
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

/// [`SolidOracle`] for a [`Union`].
pub struct UnionOracle<O>(Vec<O>);

impl<O: SolidOracle> SolidOracle for UnionOracle<O> {
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

// --------------------------------------------------------- Intersection --------

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
        self.a.signed_distance(query).min(self.b.signed_distance(query))
    }

    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        let (sa, sb) = (self.a.signed_distance(query), self.b.signed_distance(query));
        best_candidate(
            query,
            [
                self.a.project(query).map(|(p, n)| (p, n, sb >= 0.0)),
                self.b.project(query).map(|(p, n)| (p, n, sa >= 0.0)),
            ]
            .into_iter()
            .flatten(),
        )
    }
}

// ----------------------------------------------------------- Difference --------

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
        let outside = self.outer.signed_distance(query);
        let carved = self.inner.signed_distance(query);
        best_candidate(
            query,
            [
                // Outer wall survives outside the carved region.
                self.outer.project(query).map(|(p, n)| (p, n, carved <= 0.0)),
                // Cavity wall: inner's surface where it lies within `outer`, its
                // normal flipped so it still points out of the solid.
                self.inner
                    .project(query)
                    .map(|(p, n)| (p, flip(n), outside >= 0.0)),
            ]
            .into_iter()
            .flatten(),
        )
    }
}

// ---------------------------------------------------------------- shared -------

/// Among `(point, normal, on_surface)` candidates, the nearest to `query` that
/// actually lies on the combined surface, or the nearest of all as a fallback.
fn best_candidate<I>(query: &Coordinate<D>, candidates: I) -> Option<(Coordinate<D>, Direction<D>)>
where
    I: IntoIterator<Item = (Coordinate<D>, Direction<D>, bool)>,
{
    let mut valid: Option<(Scalar, Coordinate<D>, Direction<D>)> = None;
    let mut any: Option<(Scalar, Coordinate<D>, Direction<D>)> = None;
    for (point, normal, on_surface) in candidates {
        let distance = (0..D)
            .map(|k| (point[k].value() - query[k].value()).powi(2))
            .sum::<Scalar>();
        if on_surface && valid.as_ref().is_none_or(|(best, ..)| distance < *best) {
            valid = Some((distance, point.clone(), normal.clone()));
        }
        if any.as_ref().is_none_or(|(best, ..)| distance < *best) {
            any = Some((distance, point, normal));
        }
    }
    valid.or(any).map(|(_, point, normal)| (point, normal))
}

/// `normal` reversed.
fn flip(normal: Direction<D>) -> Direction<D> {
    Direction::const_from(from_fn(|k| -normal[k].value()))
}
