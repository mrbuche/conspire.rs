//! Boolean combinations of solids. Each combinator is itself a
//! [`Solid`](crate::geometry::solid::Solid), so `Difference::new(box, pores)`
//! meshes through the same driver as a primitive.

mod difference;
mod intersection;
mod union;
mod union_all;

pub use difference::{Difference, DifferenceOracle};
pub use intersection::{Intersection, IntersectionOracle};
pub use union::{Union, UnionOracle};
pub use union_all::{UnionAll, UnionAllOracle};

use crate::{
    geometry::{Coordinate, Direction},
    math::Scalar,
};
use std::array::from_fn;

const D: usize = 3;

/// Among candidates `(point, normal, penalty)` — `penalty` how far `point` sits
/// on the wrong side of the boolean, `0` for a point genuinely on the combined
/// surface, each measured *at the candidate point*, not at `query` — the one on
/// the combined surface nearest `query`; failing that, the least-penalised, and
/// among those the nearest to `query`.
fn best_candidate<I>(query: &Coordinate<D>, candidates: I) -> Option<(Coordinate<D>, Direction<D>)>
where
    I: IntoIterator<Item = (Coordinate<D>, Direction<D>, Scalar)>,
{
    let mut valid: Option<(Scalar, Coordinate<D>, Direction<D>)> = None;
    let mut fallback: Option<(Scalar, Scalar, Coordinate<D>, Direction<D>)> = None;
    for (point, normal, penalty) in candidates {
        let distance = (0..D)
            .map(|k| (point[k].value() - query[k].value()).powi(2))
            .sum::<Scalar>();
        if penalty <= 0.0 && valid.as_ref().is_none_or(|(best, ..)| distance < *best) {
            valid = Some((distance, point.clone(), normal.clone()));
        }
        if fallback
            .as_ref()
            .is_none_or(|&(p, d, ..)| penalty < p || (penalty == p && distance < d))
        {
            fallback = Some((penalty, distance, point, normal));
        }
    }
    valid
        .map(|(_, point, normal)| (point, normal))
        .or_else(|| fallback.map(|(.., point, normal)| (point, normal)))
}

/// `normal` reversed.
fn flip(normal: Direction<D>) -> Direction<D> {
    Direction::const_from(from_fn(|k| -normal[k].value()))
}
