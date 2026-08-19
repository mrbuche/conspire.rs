#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, CoordinateList, Direction,
        bbox::{BoundingBox, BoundingBoxes, Unite},
        bvh::BoundingVolumeHierarchy,
        primitive::{Solid, Union},
    },
    math::{Quantity, Scalar, Tensor},
    units::Length,
};
use std::cell::OnceCell;

const D: usize = 3;

/// How far inside its neighbours a candidate must lie, as a fraction of the
/// union's extent, before it counts as buried rather than as shared surface.
const BURIED_TOLERANCE: Scalar = 1.0e-10;

/// How near the surface, as a fraction of the union's extent, a projection has
/// to land before it has arrived.
const PROJECTION_TOLERANCE: Scalar = 1.0e-15;

/// How many steps a projection may take before giving up on that.
const PROJECTION_STEPS: usize = 16;

/// The fraction of the union's extent a search widens by when the point lies in
/// no member's box at all, leaving nothing nearer to start from.
const SEED_FRACTION: Scalar = 16.0;

/// How many members a union carries before pruning is worth its while.
///
/// Descending the hierarchy costs a traversal and the list it gathers, some
/// hundreds of nanoseconds, where asking a cylinder its distance costs a few:
/// pruning only repays that once there are enough members to skip. Measured at
/// the crossing point, where the two cost the same.
const PRUNING_THRESHOLD: usize = 256;

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
    fn within(&self, point: &Coordinate<D>, radius: Quantity<Length>) -> Vec<usize> {
        let offset = Coordinate::from([radius.value(); D]);
        self.hierarchy()
            .overlapping(&BoundingBox::from(CoordinateList::const_from([
                point - &offset,
                point + &offset,
            ])))
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
    fn searched(&self, point: &Coordinate<D>) -> Quantity<Length> {
        let zero = Quantity::default();
        let extent = self.extent();
        let reach = self.reach(point, extent);
        let seed = (extent.maximum() - extent.minimum()).norm() / SEED_FRACTION;
        let mut radius = self.clearance(point, extent);
        loop {
            let members = self.within(point, radius);
            match members.is_empty() {
                false => {
                    let nearest = self.among(point, members);
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
    /// The nearest of the members' closest points the others leave exposed.
    fn exposed(
        &self,
        point: &Coordinate<D>,
        members: impl IntoIterator<Item = usize>,
        tolerance: Quantity<Length>,
    ) -> Option<(Quantity<Length>, Coordinate<D>, Direction<D>)> {
        members
            .into_iter()
            .filter_map(|member| {
                let (candidate, normal) = self.solids[member].closest_point(point);
                (self.signed_distance(&candidate) >= -tolerance)
                    .then(|| ((point - &candidate).norm(), candidate, normal))
            })
            .min_by(|(one, ..), (other, ..)| one.total_cmp(other))
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
        if self.solids.len() <= PRUNING_THRESHOLD {
            return self
                .solids
                .iter()
                .fold(Quantity::new(Scalar::INFINITY), |least, solid| {
                    least.min(solid.signed_distance(point))
                });
        }
        self.searched(point)
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
        let extent = self.extent();
        let diagonal = (extent.maximum() - extent.minimum()).norm();
        let tolerance = diagonal * BURIED_TOLERANCE;
        if self.solids.len() <= PRUNING_THRESHOLD {
            if let Some((_, candidate, normal)) =
                self.exposed(point, 0..self.solids.len(), tolerance)
            {
                return (candidate, normal);
            }
        } else {
            let reach = self.reach(point, extent);
            let mut radius = self.searched(point).abs().max(diagonal / SEED_FRACTION);
            loop {
                match self.exposed(point, self.within(point, radius), tolerance) {
                    Some((distance, candidate, normal))
                        if distance <= radius || radius >= reach =>
                    {
                        return (candidate, normal);
                    }
                    _ if radius >= reach => break,
                    _ => radius = (radius * 2.0).min(reach),
                }
            }
        }
        project(self, point, diagonal * PROJECTION_TOLERANCE)
    }
    fn bounding_box(&self) -> BoundingBox<D> {
        self.extent().clone()
    }
}

/// Steps a point onto the surface along the gradient of the distance to it.
///
/// The distance to a union is the distance to whichever member is nearest, so
/// that member's outward normal is the gradient, and stepping the distance
/// against it lands on the surface in one go wherever the member stays the
/// nearest one. Crossing over to another member partway costs another step
/// rather than the answer, and the steps converge quadratically.
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
        current -= &normal * depth;
    }
    (current, normal)
}
