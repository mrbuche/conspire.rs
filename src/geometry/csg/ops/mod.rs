//! Boolean combinations of solids. Each combinator is itself a
//! [`Solid`](crate::geometry::solid::Solid), so `Difference::new(box, pores)`
//! meshes through the same driver as a primitive.

mod difference;
mod intersection;
mod union;

pub use difference::{Difference, DifferenceOracle};
pub use intersection::{Intersection, IntersectionOracle};
pub use union::{Union, UnionOracle};

use crate::{
    geometry::{Coordinate, Direction},
    math::Scalar,
};
use std::array::from_fn;

const D: usize = 3;

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
