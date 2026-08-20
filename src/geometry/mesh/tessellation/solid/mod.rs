#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates, Direction, DirectionsRef,
        bbox::BoundingBox,
        bvh::BoundingVolumeHierarchy,
        mesh::tessellation::{D, Tessellation},
        primitive::Solid,
    },
    math::{Quantity, Scalar, Tensor},
    units::Length,
};
use std::thread::{available_parallelism, scope};

/// How squarely a ray must meet a facet for the crossing to be believed, since
/// one grazing the surface says nothing reliable about which side it came from.
const GRAZING_TOLERANCE: Scalar = 1.0e-4;

/// Directions deliberately off the axes, so that a ray is unlikely to run along
/// a facet or through the seam between two of them.
const DIRECTIONS: [Direction<D>; 3] = [
    Direction::const_from([1.0, 0.140_412_03, 0.092_153_88]),
    Direction::const_from([0.097_153_2, 1.0, 0.131_771_4]),
    Direction::const_from([0.123_456_7, 0.087_654_3, 1.0]),
];

/// A closed tessellation stands for the volume it encloses, answering for that
/// volume the same questions a solid described in closed form answers.
///
/// The sign comes from casting a ray and reading the facet it first meets:
/// meeting its back is leaving the volume, so the point was inside. A ray
/// grazing a facet cannot say, and another direction is tried instead. The
/// magnitude comes from the nearest point on the surface, which the hierarchy
/// finds without regard to the sign.
impl Solid<D> for Tessellation {
    fn signed_distance(&self, point: &Coordinate<D>) -> Quantity<Length> {
        Surface::of(self).distance_at(point)
    }
    /// Gathers the surface once and hands the points out across the threads,
    /// rather than paying for either at every point.
    fn signed_distances(&self, points: &Coordinates<D>) -> Vec<Quantity<Length>> {
        let surface = Surface::of(self);
        let number_of_points = points.len();
        let mut distances = vec![Quantity::new(Scalar::INFINITY); number_of_points];
        let threads = available_parallelism().map_or(1, |threads| threads.get());
        let chunk_size = number_of_points.div_ceil(threads).max(1);
        scope(|scope| {
            let surface = &surface;
            distances
                .chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk, distances)| {
                    scope.spawn(move || {
                        let offset = chunk * chunk_size;
                        distances
                            .iter_mut()
                            .enumerate()
                            .for_each(|(local, distance)| {
                                *distance = surface.distance_at(&points[offset + local])
                            })
                    });
                });
        });
        distances
    }
    fn closest_point(&self, point: &Coordinate<D>) -> (Coordinate<D>, Direction<D>) {
        Surface::of(self).closest_at(point)
    }
    /// Gathers the surface once and hands the points out across the threads,
    /// as [`signed_distances`](Solid::signed_distances) does.
    fn closest_points(&self, points: &Coordinates<D>) -> Vec<(Coordinate<D>, Direction<D>)> {
        let surface = Surface::of(self);
        let number_of_points = points.len();
        let mut closest = vec![None; number_of_points];
        let threads = available_parallelism().map_or(1, |threads| threads.get());
        let chunk_size = number_of_points.div_ceil(threads).max(1);
        scope(|scope| {
            let surface = &surface;
            closest
                .chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk, closest)| {
                    scope.spawn(move || {
                        let offset = chunk * chunk_size;
                        closest.iter_mut().enumerate().for_each(|(local, closest)| {
                            *closest = Some(surface.closest_at(&points[offset + local]))
                        })
                    });
                });
        });
        closest.into_iter().flatten().collect()
    }
    /// A tessellation of no facets encloses nothing.
    fn is_empty(&self) -> bool {
        self.mesh().connectivities().iter().flatten().count() == 0
    }
    fn bounding_box(&self) -> BoundingBox<D> {
        BoundingBox::from(self.mesh().coordinates().clone())
    }
}

/// A tessellation's surface gathered for querying, holding nothing but borrows
/// so that the threads of one query may share it.
struct Surface<'a> {
    bvh: &'a BoundingVolumeHierarchy<D>,
    coordinates: &'a Coordinates<D>,
    directions: [Direction<D>; 3],
    elements: Vec<&'a [usize]>,
    normals: DirectionsRef<'a, D>,
}

impl<'a> Surface<'a> {
    fn of(tessellation: &'a Tessellation) -> Self {
        Self {
            bvh: tessellation.bvh(),
            coordinates: tessellation.mesh().coordinates(),
            directions: DIRECTIONS.map(|direction| direction.normalized()),
            elements: tessellation
                .mesh()
                .connectivities()
                .iter()
                .flatten()
                .collect(),
            normals: tessellation.normals().iter().flatten().collect(),
        }
    }
    /// Negative within, as a distance to a solid is, which is the opposite of
    /// the sense the ray casting itself reads off.
    ///
    /// A point the hierarchy finds no surface for is left as far outside as it
    /// can be, so that whatever asked discards it.
    fn distance_at(&self, point: &Coordinate<D>) -> Quantity<Length> {
        let inside = self
            .directions
            .iter()
            .find_map(|direction| {
                let ray = (point.clone(), direction.clone()).into();
                match self.bvh.intersect(&ray, self.coordinates, &self.elements) {
                    None => Some(false),
                    Some(hit) => {
                        let normal = &self.normals[hit.index()];
                        let cosine = (direction * normal) / normal.norm();
                        (cosine.abs() > GRAZING_TOLERANCE).then_some(cosine > 0.0)
                    }
                }
            })
            .unwrap_or(false);
        match self
            .bvh
            .closest_point(point, self.coordinates, &self.elements)
        {
            Some((closest, _)) => {
                let magnitude = (&closest - point).norm();
                if inside { -magnitude } else { magnitude }
            }
            None => Quantity::new(Scalar::INFINITY),
        }
    }
    /// A point with no surface to find has nowhere to go, and stays put.
    fn closest_at(&self, point: &Coordinate<D>) -> (Coordinate<D>, Direction<D>) {
        match self
            .bvh
            .closest_point(point, self.coordinates, &self.elements)
        {
            Some((closest, facet)) => (closest, self.normals[facet].clone().normalized()),
            None => (point.clone(), self.directions[0].clone()),
        }
    }
}
