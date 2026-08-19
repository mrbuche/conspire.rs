#[cfg(test)]
pub mod test;

mod cylinder;
mod unite;

use crate::{
    geometry::{Coordinate, Direction, bbox::BoundingBox},
    math::Quantity,
    units::Length,
};

/// A closed region of space, described by where it is rather than by a mesh of
/// its boundary.
///
/// Meshing consumes geometry through point queries alone — whether a node is
/// inside, how far it sits from the surface, and where on the surface it
/// projects to — so a primitive answering those in closed form serves the
/// pipeline without ever being tessellated, and answers exactly where a
/// tessellation only approximates.
pub trait Solid<const D: usize> {
    /// The distance from a point to the boundary, negative inside.
    fn signed_distance(&self, point: &Coordinate<D>) -> Quantity<Length>;
    /// The closest point on the boundary, and the outward normal there.
    fn closest_point(&self, point: &Coordinate<D>) -> (Coordinate<D>, Direction<D>);
    /// The axis-aligned box containing the solid.
    fn bounding_box(&self) -> BoundingBox<D>;
    /// Whether a point lies within the solid, its boundary included.
    fn contains(&self, point: &Coordinate<D>) -> bool {
        self.signed_distance(point) <= Quantity::default()
    }
}

/// A circular cylinder with flat caps, spanning two endpoints.
#[derive(Clone, Debug)]
pub struct Cylinder {
    base: Coordinate<3>,
    axis: Direction<3>,
    height: Quantity<Length>,
    radius: Quantity<Length>,
}

/// The union of solids, being the region covered by any of them.
///
/// Held as a flat list rather than a tree of binary combinators so that the
/// members stay one homogeneous collection, which keeps queries free of
/// dynamic dispatch and leaves room to prune them against a hierarchy of the
/// members' bounding boxes.
#[derive(Clone, Debug)]
pub struct Union<S> {
    solids: Vec<S>,
}
