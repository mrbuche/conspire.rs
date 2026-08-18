#[cfg(test)]
mod test;

use std::{
    f64::consts::TAU,
    thread::{available_parallelism, scope},
};

use crate::{
    geometry::{Direction, DirectionsRef, mesh::tessellation::Tessellation},
    math::{Quantity, QuantityVector, Scalar, Tensor},
    units::Length,
};

impl Tessellation {
    /// Estimates local thickness at each node via the shape diameter function.[^1]
    ///
    /// [^1]: L. Shapira, A. Shamir, and D. Cohen-Or, [Visual Comp. **24**, 249 (2008)](https://doi.org/10.1007/s00371-007-0197-5).
    pub fn shape_diameter_function(
        &self,
        half_angle: Scalar,
        rings: usize,
        azimuthal: usize,
    ) -> QuantityVector<Length> {
        let mesh = self.mesh();
        let bvh = self.bvh();
        let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
        let coordinates = mesh.coordinates();
        let centroids = mesh.centroids();
        let normals: DirectionsRef<'_, 3> = self.normals.iter().flatten().collect();
        let number_of_faces = normals.len();
        let mut face_diameters = vec![Quantity::<Length>::default(); number_of_faces];
        let threads = available_parallelism().map_or(1, |threads| threads.get());
        let chunk_size = number_of_faces.div_ceil(threads).max(1);
        scope(|scope| {
            let (bvh, elements, centroids, normals) = (bvh, &elements, &centroids, &normals);
            face_diameters
                .chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk, diameters)| {
                    scope.spawn(move || {
                        let offset = chunk * chunk_size;
                        diameters
                            .iter_mut()
                            .enumerate()
                            .for_each(|(local, diameter)| {
                                let face = offset + local;
                                let samples =
                                    cone_directions(&-&normals[face], half_angle, rings, azimuthal)
                                        .into_iter()
                                        .filter_map(|(direction, weight)| {
                                            let ray = (centroids[face].clone(), direction).into();
                                            bvh.intersect(&ray, coordinates, elements)
                                                .filter(|hit| hit.index() != face)
                                                .map(|hit| (hit.distance(), weight))
                                        })
                                        .collect();
                                *diameter = weighted_diameter(samples);
                            });
                    });
                });
        });
        interpolate_to_nodes(face_diameters.into(), elements, coordinates.len())
    }
}

fn interpolate_to_nodes(
    face_diameters: QuantityVector<Length>,
    elements: Vec<&[usize]>,
    number_of_nodes: usize,
) -> QuantityVector<Length> {
    let mut nodal = QuantityVector::zero(number_of_nodes);
    let mut counts = vec![0; number_of_nodes];
    elements
        .into_iter()
        .zip(face_diameters)
        .for_each(|(element, diameter)| {
            element.iter().for_each(|&node| {
                nodal[node] += diameter;
                counts[node] += 1;
            })
        });
    nodal.iter_mut().zip(counts).for_each(|(value, count)| {
        if count > 0 {
            *value /= count as Scalar
        }
    });
    nodal
}

fn cone_directions(
    axis: &Direction<3>,
    half_angle: Scalar,
    rings: usize,
    azimuthal: usize,
) -> Vec<(Direction<3>, Scalar)> {
    let basis = axis.orthonormal_basis();
    let (axis, tangent_1, tangent_2) = (&basis[0], &basis[1], &basis[2]);
    let sigma = half_angle / 3.0;
    let mut directions = Vec::with_capacity(1 + rings * azimuthal);
    directions.push((axis.clone(), 1.0));
    for ring in 1..=rings {
        let polar = half_angle * ring as Scalar / rings as Scalar;
        let (sin_polar, cos_polar) = polar.sin_cos();
        let weight = (-0.5 * (polar / sigma).powi(2)).exp();
        for sample in 0..azimuthal {
            let (sin_azimuth, cos_azimuth) =
                (TAU * sample as Scalar / azimuthal as Scalar).sin_cos();
            let direction = axis * cos_polar
                + tangent_1 * (sin_polar * cos_azimuth)
                + tangent_2 * (sin_polar * sin_azimuth);
            directions.push((direction, weight));
        }
    }
    directions
}

fn weighted_diameter(samples: Vec<(Quantity<Length>, Scalar)>) -> Quantity<Length> {
    if samples.is_empty() {
        return Quantity::default();
    }
    let mut distances: Vec<Quantity<Length>> =
        samples.iter().map(|&(distance, _)| distance).collect();
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = distances[distances.len() / 2];
    let mean = distances.iter().copied().sum::<Quantity<Length>>() / distances.len() as Scalar;
    let standard_deviation = Quantity::new(
        (distances
            .iter()
            .map(|&distance| (distance - mean).value().powi(2))
            .sum::<Scalar>()
            / distances.len() as Scalar)
            .sqrt(),
    );
    let (numerator, denominator) = samples
        .into_iter()
        .filter(|&(distance, _)| (distance - median).abs() <= standard_deviation)
        .fold(
            (Quantity::default(), 0.0),
            |(numerator, denominator), (distance, weight)| {
                (numerator + distance * weight, denominator + weight)
            },
        );
    if denominator > 0.0 {
        numerator / denominator
    } else {
        median
    }
}
