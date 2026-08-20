#[cfg(test)]
pub mod test;

mod cylinder;
mod unite;

use crate::{
    geometry::{
        Coordinate, Coordinates, Direction, bbox::BoundingBox, bvh::BoundingVolumeHierarchy,
    },
    math::{Quantity, Tensor},
    units::Length,
};
use std::cell::OnceCell;

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
    /// The distances at many points at once.
    ///
    /// Meshing asks by the meshful rather than by the point, and a solid whose
    /// answer costs something to set up — gathering a surface's elements, or
    /// handing the points out across threads — wants that cost paid once for
    /// the lot instead of once apiece. One at a time by default, for the solid
    /// with nothing to set up.
    fn signed_distances(&self, points: &Coordinates<D>) -> Vec<Quantity<Length>> {
        points
            .iter()
            .map(|point| self.signed_distance(point))
            .collect()
    }
    /// The closest points at many points at once, for the reason
    /// [`signed_distances`](Solid::signed_distances) takes many at once.
    fn closest_points(&self, points: &Coordinates<D>) -> Vec<(Coordinate<D>, Direction<D>)> {
        points
            .iter()
            .map(|point| self.closest_point(point))
            .collect()
    }
    /// Whether the solid encloses nothing at all, and so has no surface with
    /// which to answer.
    ///
    /// Only a solid described by a surface can be given none; one described by
    /// where it is always encloses somewhere.
    fn is_empty(&self) -> bool {
        false
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
/// dynamic dispatch and lets a hierarchy over their bounding boxes prune them
/// all at once rather than a level at a time.
///
/// Three-dimensional alone, since that hierarchy is, though [`Solid`] itself is
/// not; a union in another dimension can generalize the pruning when something
/// needs one.
pub struct Union<S> {
    solids: Vec<S>,
    hierarchy: OnceCell<BoundingVolumeHierarchy<3>>,
    extent: OnceCell<BoundingBox<3>>,
}
