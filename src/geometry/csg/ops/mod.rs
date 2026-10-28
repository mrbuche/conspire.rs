//! Boolean combinations of solids. Each combinator is itself a
//! [`SolidOracle`](crate::geometry::solid::SolidOracle), so they nest.
//!
//! A combinator's closest boundary point is the nearest of its operands' closest points that
//! lies on the boundary of the combination. It is always a boundary point, and the true nearest
//! one except possibly near a crease between two surfaces, where a hidden candidate may have
//! been nearer. It is `None` only if no candidate lies on the boundary.

#[cfg(test)]
mod test;

mod difference;
mod intersection;
mod union;
mod union_all;

pub use difference::Difference;
pub use intersection::Intersection;
pub use union::Union;
pub use union_all::UnionAll;

use crate::{
    geometry::{Coordinate, Direction},
    math::Tensor,
};

type Candidate = Option<(Coordinate<3>, Direction<3>, bool)>;

fn nearest_surviving(
    query: &Coordinate<3>,
    candidates: impl IntoIterator<Item = Candidate>,
) -> Option<(Coordinate<3>, Direction<3>)> {
    candidates
        .into_iter()
        .flatten()
        .filter(|(.., survives)| *survives)
        .min_by(|(a, ..), (b, ..)| (a - query).norm().total_cmp(&(b - query).norm()))
        .map(|(point, normal, _)| (point, normal))
}
