#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Direction, DirectionsRef,
        mesh::{
            Mesh,
            tessellation::{D, Tessellation},
        },
    },
    math::{Scalar, Tensor},
};
use std::thread::{available_parallelism, scope};

const GRAZING_TOLERANCE: Scalar = 1.0e-4;
const TRIM_RATIO: Scalar = 0.1;
const DIRECTIONS: [Direction<D>; 3] = [
    Direction::const_from([1.0, 0.140_412_03, 0.092_153_88]),
    Direction::const_from([0.097_153_2, 1.0, 0.131_771_4]),
    Direction::const_from([0.123_456_7, 0.087_654_3, 1.0]),
];

impl Tessellation {
    /// Discards the cells of a background mesh lying outside this
    /// tessellation, leaving a mesh that covers the volume it encloses.
    ///
    /// A cell survives when the signed distances at its nodes satisfy
    /// `minimum + 0.1 * maximum >= 0`, so the cells straddling the surface
    /// are kept for [`buffer`](Mesh::buffer) to fit onto it.
    pub fn trim(&self, mesh: &mut Mesh<D>) -> Result<(), &'static str> {
        let bvh = self.bvh();
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let normals: DirectionsRef<'_, D> = self.normals().iter().flatten().collect();
        let directions = DIRECTIONS.map(|direction| direction.normalized());
        let coordinates = mesh.coordinates();
        let number_of_nodes = coordinates.len();
        let mut signed = vec![Scalar::NEG_INFINITY; number_of_nodes];
        let threads = available_parallelism().map_or(1, |threads| threads.get());
        let chunk_size = number_of_nodes.div_ceil(threads).max(1);
        scope(|scope| {
            let (elements, normals, directions) = (&elements, &normals, &directions);
            signed
                .chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk, distances)| {
                    scope.spawn(move || {
                        let offset = chunk * chunk_size;
                        distances
                            .iter_mut()
                            .enumerate()
                            .for_each(|(local, distance)| {
                                let point = &coordinates[offset + local];
                                let inside = directions
                                    .iter()
                                    .find_map(|direction| {
                                        let ray = (point.clone(), direction.clone()).into();
                                        match bvh.intersect(&ray, surface_coordinates, elements) {
                                            None => Some(false),
                                            Some(hit) => {
                                                let normal = &normals[hit.index()];
                                                let cosine = (direction * normal) / normal.norm();
                                                (cosine.abs() > GRAZING_TOLERANCE)
                                                    .then_some(cosine > 0.0)
                                            }
                                        }
                                    })
                                    .unwrap_or(false);
                                if let Some((closest, _)) =
                                    bvh.closest_point(point, surface_coordinates, elements)
                                {
                                    let magnitude = (&closest - point).norm().value();
                                    *distance = if inside { magnitude } else { -magnitude };
                                }
                            });
                    });
                });
        });
        mesh.keep_hexes(|_, hex, _| {
            let (minimum, maximum) = hex.iter().fold(
                (Scalar::INFINITY, Scalar::NEG_INFINITY),
                |(minimum, maximum), &node| (minimum.min(signed[node]), maximum.max(signed[node])),
            );
            minimum + TRIM_RATIO * maximum >= 0.0
        })
    }
}
