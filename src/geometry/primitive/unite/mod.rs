#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, CoordinateList, Coordinates, Direction,
        bbox::{BoundingBox, BoundingBoxes, Unite},
        bvh::BoundingVolumeHierarchy,
        primitive::{Solid, Union},
    },
    math::{Quantity, Scalar, Tensor},
    units::Length,
};
use std::cell::OnceCell;

const D: usize = 3;

/// How near the surface, as a fraction of the union's extent, a projection has
/// to land before it has arrived.
const PROJECTION_TOLERANCE: Scalar = 1.0e-15;

/// How many steps a projection may take before giving up on that.
const PROJECTION_STEPS: usize = 64;

/// The shortest fraction of a step worth trying, below which halving it again
/// buys nothing.
const SHORTEST_STEP: Scalar = 1.0e-4;

/// The fraction of the union's extent a search widens by when the point lies in
/// no member's box at all, leaving nothing nearer to start from.
const SEED_FRACTION: Scalar = 16.0;

/// How many members a union carries before pruning is worth its while.
///
/// Descending the hierarchy costs a traversal, where asking a cylinder its
/// distance costs a few nanoseconds, so pruning only repays itself once there
/// are enough members to skip. Measured at the crossing point, where the two
/// cost the same: pruning breaks even at this many and wins from there, by
/// nearly twice at twice as many and by sevenfold at sixteen times as many.
const PRUNING_THRESHOLD: usize = 32;

impl<S> Union<S> {
    /// Unites solids into the region any of them covers.
    pub fn new(solids: Vec<S>) -> Self {
        assert!(!solids.is_empty(), "a union needs at least one solid");
        Self {
            solids,
            hierarchy: OnceCell::new(),
            extent: OnceCell::new(),
        }
    }
    pub fn solids(&self) -> &[S] {
        &self.solids
    }
}

impl<S: Solid<D>> Union<S> {
    /// The hierarchy over the members' boxes, built when first asked for.
    fn hierarchy(&self) -> &BoundingVolumeHierarchy<D> {
        self.hierarchy.get_or_init(|| {
            self.solids
                .iter()
                .map(|solid| solid.bounding_box())
                .collect::<BoundingBoxes<D>>()
                .into()
        })
    }
    /// The members whose boxes reach within a radius of a point.
    ///
    /// A member left out is farther than the radius, since its box misses a cube
    /// of that half-width, which is what lets a search stop once it holds an
    /// answer nearer than the radius it searched.
    fn within(&self, point: &Coordinate<D>, radius: Quantity<Length>, found: &mut Vec<usize>) {
        let offset = Coordinate::from([radius.value(); D]);
        self.hierarchy().overlapping_into(
            &BoundingBox::from(CoordinateList::const_from([
                point - &offset,
                point + &offset,
            ])),
            found,
        )
    }
    /// The radius covering every member, whatever the point.
    fn reach(&self, point: &Coordinate<D>, extent: &BoundingBox<D>) -> Quantity<Length> {
        (point - extent.minimum())
            .norm_inf()
            .max((point - extent.maximum()).norm_inf())
    }
    /// The radius at which a search first reaches the union at all, being zero
    /// for a point within its box.
    ///
    /// Starting a search here rather than at nothing spares a point out beyond
    /// the union the whole ladder of doublings it would otherwise climb before
    /// its query box met anything.
    fn clearance(&self, point: &Coordinate<D>, extent: &BoundingBox<D>) -> Quantity<Length> {
        (0..D).fold(Quantity::default(), |clearance, axis| {
            clearance
                .max(extent.minimum()[axis] - point[axis])
                .max(point[axis] - extent.maximum()[axis])
        })
    }
    /// The distance to the nearest member, pruning away those too far to matter.
    ///
    /// Only the members boxed around the point can hold it, so a negative answer
    /// is settled by the first query. A positive one bounds how far the search
    /// has to reach, and re-querying at that bound either confirms it or
    /// tightens it, until what was found is nearer than what was searched.
    fn searched(&self, point: &Coordinate<D>, found: &mut Vec<usize>) -> Quantity<Length> {
        let zero = Quantity::default();
        let extent = self.extent();
        let reach = self.reach(point, extent);
        let seed = (extent.maximum() - extent.minimum()).norm() / SEED_FRACTION;
        let mut radius = self.clearance(point, extent);
        loop {
            self.within(point, radius, found);
            match found.is_empty() {
                false => {
                    let nearest = self.among(point, found.iter().copied());
                    if nearest <= zero || nearest <= radius || radius >= reach {
                        return nearest;
                    }
                    radius = nearest
                }
                true if radius >= reach => unreachable!("a union covers at least one solid"),
                true => radius = (radius * 2.0).max(seed).min(reach),
            }
        }
    }
    /// The distance to the nearest of the members named.
    ///
    /// Folded through a minimum rather than tracked alongside which member won
    /// it, which would cost a branch per member where a minimum costs none, and
    /// this runs once per member of every query there is.
    fn among(
        &self,
        point: &Coordinate<D>,
        members: impl IntoIterator<Item = usize>,
    ) -> Quantity<Length> {
        members
            .into_iter()
            .fold(Quantity::new(Scalar::INFINITY), |least, member| {
                least.min(self.solids[member].signed_distance(point))
            })
    }
    /// Which member is the nearest, for the rare query needing more of it than
    /// how far off it is.
    fn nearest(&self, point: &Coordinate<D>) -> usize {
        let mut nearest = None;
        self.solids.iter().enumerate().for_each(|(member, solid)| {
            let distance = solid.signed_distance(point);
            if nearest.is_none_or(|(least, _)| distance < least) {
                nearest = Some((distance, member))
            }
        });
        nearest.expect("a union needs at least one solid").1
    }
    /// The distance to the nearest member, gathering into a list the caller
    /// holds onto so that a query per point does not cost a list per point.
    fn distance_with(&self, point: &Coordinate<D>, found: &mut Vec<usize>) -> Quantity<Length> {
        if self.solids.len() <= PRUNING_THRESHOLD {
            self.solids
                .iter()
                .fold(Quantity::new(Scalar::INFINITY), |least, solid| {
                    least.min(solid.signed_distance(point))
                })
        } else {
            self.searched(point, found)
        }
    }
    /// The union's box, held onto rather than gathered again for every query.
    fn extent(&self) -> &BoundingBox<D> {
        self.extent.get_or_init(|| {
            let mut boxes = self.solids.iter().map(|solid| solid.bounding_box());
            let first = boxes.next().expect("a union needs at least one solid");
            boxes.fold(first, |united, other| united.unite(other))
        })
    }
}

impl<S: Solid<D>> Solid<D> for Union<S> {
    /// The smallest of the members' distances.
    ///
    /// Outside the union this is exact, since the nearest surface of any member
    /// is the nearest surface of them all. Inside it holds the right sign but
    /// understates the depth, reporting the depth within a single member where
    /// the union may hold the point deeper still.
    fn signed_distance(&self, point: &Coordinate<D>) -> Quantity<Length> {
        self.distance_with(point, &mut Vec::new())
    }
    /// Gathers what a query passes over into one list for the whole meshful,
    /// rather than into one per point.
    fn signed_distances(&self, points: &Coordinates<D>) -> Vec<Quantity<Length>> {
        let mut found = Vec::new();
        points
            .iter()
            .map(|point| self.distance_with(point, &mut found))
            .collect()
    }
    /// The nearest point of the surface, reached by walking down the distance.
    ///
    /// Outside the union this is exact and immediate: the distance is the
    /// distance to the nearest member, its gradient is that member's outward
    /// normal, and stepping the one against the other lands on the very point
    /// that measured it.
    ///
    /// Choosing among the members' own closest points instead would be exact
    /// outside too, but not within. A member whose nearest point is buried is
    /// not thereby a distant member — the rest of its surface may still be the
    /// nearest thing there is — and discarding it wholesale sends the answer
    /// clear across the solid to whichever member kept an exposed point.
    /// Walking down the gradient cannot leave the neighbourhood it starts in.
    fn closest_point(&self, point: &Coordinate<D>) -> (Coordinate<D>, Direction<D>) {
        let extent = self.extent();
        let diagonal = (extent.maximum() - extent.minimum()).norm();
        project(self, point, diagonal * PROJECTION_TOLERANCE)
    }
    fn bounding_box(&self) -> BoundingBox<D> {
        self.extent().clone()
    }
}

/// Steps a point onto the surface along the gradient of the distance to it.
///
/// A step lands on the surface outright wherever one member stays the nearest
/// throughout it. Where two members trade places partway — along a crease
/// where they meet — a full step overshoots onto the other's far side and the
/// next step overshoots back, so a step that fails to close the distance is
/// halved until it does. Left to stride, the two would trade blows without
/// ever arriving.
fn project<S: Solid<D>>(
    union: &Union<S>,
    point: &Coordinate<D>,
    tolerance: Quantity<Length>,
) -> (Coordinate<D>, Direction<D>) {
    let mut current = point.clone();
    let mut normal = union.solids[union.nearest(&current)]
        .closest_point(&current)
        .1;
    for _ in 0..PROJECTION_STEPS {
        let depth = union.signed_distance(&current);
        normal = union.solids[union.nearest(&current)]
            .closest_point(&current)
            .1;
        if depth.abs() <= tolerance {
            break;
        }
        let mut fraction = 1.0;
        loop {
            let trial = &current - &normal * (depth * fraction);
            if union.signed_distance(&trial).abs() < depth.abs() || fraction <= SHORTEST_STEP {
                current = trial;
                break;
            }
            fraction *= 0.5
        }
    }
    (current, normal)
}
