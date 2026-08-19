#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Direction,
        bbox::{BoundingBox, Unite},
        primitive::{Solid, Union},
    },
    math::{Quantity, Scalar, Tensor},
    units::Length,
};

/// How far inside its neighbours a candidate must lie, as a fraction of the
/// union's extent, before it counts as buried rather than as shared surface.
const BURIED_TOLERANCE: Scalar = 1.0e-10;

/// How near the surface, as a fraction of the union's extent, a projection has
/// to land before it has arrived.
const PROJECTION_TOLERANCE: Scalar = 1.0e-15;

/// How many steps a projection may take before giving up on that.
const PROJECTION_STEPS: usize = 16;

impl<S> Union<S> {
    /// Unites solids into the region any of them covers.
    pub fn new(solids: Vec<S>) -> Self {
        assert!(!solids.is_empty(), "a union needs at least one solid");
        Self { solids }
    }
    pub fn solids(&self) -> &[S] {
        &self.solids
    }
}

impl<const D: usize, S: Solid<D>> Solid<D> for Union<S> {
    /// The smallest of the members' distances.
    ///
    /// Outside the union this is exact, since the nearest surface of any member
    /// is the nearest surface of them all. Inside it holds the right sign but
    /// understates the depth, reporting the depth within a single member where
    /// the union may hold the point deeper still.
    fn signed_distance(&self, point: &Coordinate<D>) -> Quantity<Length> {
        self.solids
            .iter()
            .map(|solid| solid.signed_distance(point))
            .fold(Quantity::new(Scalar::INFINITY), Quantity::min)
    }
    /// The nearest of the members' closest points that the others leave exposed,
    /// or a projection onto the surface where they leave none.
    ///
    /// A member's closest point is only the union's if the other members do not
    /// bury it, since the surface they overlap is interior and no longer
    /// boundary. Outside the union no candidate is ever buried — burying one
    /// would put the burying member nearer than the member owning it, which is
    /// a contradiction — so the nearest candidate is exact there.
    ///
    /// Inside a crossing every candidate can be buried at once, each member's
    /// nearest surface lying within another, and then there is no candidate to
    /// take. Walking down the distance's own gradient reaches the surface
    /// regardless, so that case projects rather than choosing.
    fn closest_point(&self, point: &Coordinate<D>) -> (Coordinate<D>, Direction<D>) {
        let extent = self.bounding_box();
        let diagonal = (extent.maximum() - extent.minimum()).norm();
        let tolerance = diagonal * BURIED_TOLERANCE;
        let mut exposed: Option<(Quantity<Length>, Coordinate<D>, Direction<D>)> = None;
        self.solids.iter().for_each(|solid| {
            let (candidate, normal) = solid.closest_point(point);
            if self.signed_distance(&candidate) >= -tolerance {
                let distance = (point - &candidate).norm();
                if exposed
                    .as_ref()
                    .is_none_or(|(nearest, _, _)| distance < *nearest)
                {
                    exposed = Some((distance, candidate, normal))
                }
            }
        });
        match exposed {
            Some((_, candidate, normal)) => (candidate, normal),
            None => project(self, point, diagonal * PROJECTION_TOLERANCE),
        }
    }
    fn bounding_box(&self) -> BoundingBox<D> {
        let mut boxes = self.solids.iter().map(|solid| solid.bounding_box());
        let first = boxes.next().expect("a union needs at least one solid");
        boxes.fold(first, |united, other| united.unite(other))
    }
}

/// Steps a point onto the surface along the gradient of the distance to it.
///
/// The distance to a union is the distance to whichever member is nearest, so
/// that member's outward normal is the gradient, and stepping the distance
/// against it lands on the surface in one go wherever the member stays the
/// nearest one. Crossing over to another member partway costs another step
/// rather than the answer, and the steps converge quadratically.
fn project<const D: usize, S: Solid<D>>(
    union: &Union<S>,
    point: &Coordinate<D>,
    tolerance: Quantity<Length>,
) -> (Coordinate<D>, Direction<D>) {
    let mut current = point.clone();
    let mut normal = union.solids[0].closest_point(&current).1;
    for _ in 0..PROJECTION_STEPS {
        let (depth, nearest) = union.solids.iter().enumerate().fold(
            (Quantity::new(Scalar::INFINITY), 0),
            |(depth, nearest), (index, solid)| {
                let distance = solid.signed_distance(&current);
                if distance < depth {
                    (distance, index)
                } else {
                    (depth, nearest)
                }
            },
        );
        normal = union.solids[nearest].closest_point(&current).1;
        if depth.abs() <= tolerance {
            break;
        }
        current -= &normal * depth;
    }
    (current, normal)
}
