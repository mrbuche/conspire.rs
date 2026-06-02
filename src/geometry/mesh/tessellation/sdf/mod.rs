#[cfg(test)]
mod test;

use std::f64::consts::TAU;

use crate::{
    geometry::{Coordinate, bvh::BoundingVolumeHierarchy, mesh::tessellation::Tessellation},
    math::{Scalar, Tensor},
};

impl Tessellation {
    pub fn shape_diameter_function(
        &self,
        half_angle: Scalar,
        rings: usize,
        azimuthal: usize,
    ) -> Vec<Scalar> {
        let mesh = self.mesh();
        let bvh = BoundingVolumeHierarchy::from(mesh);
        let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
        let coordinates = mesh.coordinates();
        mesh.centroids()
            .iter()
            .zip(self.normals.iter().flatten())
            .enumerate()
            .map(|(face, (centroid, normal))| {
                let samples = cone_directions(&-normal, half_angle, rings, azimuthal)
                    .into_iter()
                    .filter_map(|(direction, weight)| {
                        let ray = (centroid.clone(), direction).into();
                        bvh.intersect(&ray, coordinates, &elements)
                            .filter(|hit| hit.index() != face)
                            .map(|hit| (hit.distance(), weight))
                    })
                    .collect();
                weighted_diameter(samples)
            })
            .collect()
    }
}

fn cone_directions(
    axis: &Coordinate<3>,
    half_angle: Scalar,
    rings: usize,
    azimuthal: usize,
) -> Vec<(Coordinate<3>, Scalar)> {
    let basis = axis.orthonormal_basis();
    let (axis, tangent_1, tangent_2) = (&basis[0], &basis[1], &basis[2]);
    let mut directions = Vec::with_capacity(1 + rings * azimuthal);
    directions.push((axis.clone(), 1.0));
    for ring in 1..=rings {
        let polar = half_angle * ring as Scalar / rings as Scalar;
        let (sin_polar, cos_polar) = polar.sin_cos();
        for sample in 0..azimuthal {
            let (sin_azimuth, cos_azimuth) =
                (TAU * sample as Scalar / azimuthal as Scalar).sin_cos();
            let direction = axis * cos_polar
                + tangent_1 * (sin_polar * cos_azimuth)
                + tangent_2 * (sin_polar * sin_azimuth);
            directions.push((direction, cos_polar));
        }
    }
    directions
}

fn weighted_diameter(samples: Vec<(Scalar, Scalar)>) -> Scalar {
    if samples.is_empty() {
        return 0.0;
    }
    let mut distances: Vec<Scalar> = samples.iter().map(|&(distance, _)| distance).collect();
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = distances[distances.len() / 2];
    let mean = distances.iter().sum::<Scalar>() / distances.len() as Scalar;
    let standard_deviation = (distances
        .iter()
        .map(|distance| (distance - mean).powi(2))
        .sum::<Scalar>()
        / distances.len() as Scalar)
        .sqrt();
    let (numerator, denominator) = samples
        .iter()
        .filter(|&&(distance, _)| (distance - median).abs() <= standard_deviation)
        .fold(
            (0.0, 0.0),
            |(numerator, denominator), &(distance, weight)| {
                (numerator + weight * distance, denominator + weight)
            },
        );
    if denominator > 0.0 {
        numerator / denominator
    } else {
        median
    }
}
