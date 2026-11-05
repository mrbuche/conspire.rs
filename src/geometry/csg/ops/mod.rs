//! Boolean combinations of solids. Each combinator is itself a
//! [`SolidOracle`](crate::geometry::solid::SolidOracle), so they nest.
//!
//! A combinator's closest boundary point is the nearest of its operands' closest points that
//! lies on the boundary of the combination. It is always a boundary point, and is marked exact
//! when it is also the nearest candidate overall; otherwise a hidden candidate was nearer and
//! the true closest point may be on a crease. It is `None` only if no candidate lies on the
//! boundary.

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
