#[cfg(test)]
mod test;

pub mod curvature;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{
            Tessellation,
            differential::sizing::{Creases, Unresolved, sizing_field},
        },
        ntree::{node::cell::Cell, sizing::curvature::CurvatureSizing},
    },
    math::{Quantity, Scalar, Tensor, TensorVec},
    units::Length,
};
use std::{array::from_fn, f64::consts::FRAC_PI_4};

const D: usize = 3;

/// Parameters controlling octree refinement.
pub struct Sizing<'a> {
    pub(crate) center: Coordinate<D>,
    pub(crate) coordinates: &'a Coordinates<D>,
    pub(crate) elements: Vec<&'a [usize]>,
    pub(crate) levels: u32,
    pub(crate) min_length: Quantity<Length>,
    pub(crate) scale: Scalar,
    pub(crate) targets: Vec<Quantity<Length>>,
}

impl<'a> Sizing<'a> {
    pub(crate) fn fits<T: Cell>(&self) -> bool {
        1_usize
            .checked_shl(self.levels)
            .and_then(T::length)
            .is_some()
    }
    pub fn levels(&self) -> u32 {
        self.levels
    }
    pub fn new(
        tessellation: &'a Tessellation,
        scale: Scalar,
        curvature: CurvatureSizing,
        padding: u16,
    ) -> Self {
        let CurvatureSizing {
            tolerance,
            gradation,
            floor_fraction,
        } = curvature;
        let sdf = tessellation.shape_diameter_function(FRAC_PI_4, 3, 10);
        let coordinates = tessellation.mesh().coordinates();
        if coordinates.is_empty() {
            return Self {
                center: Coordinate::const_from([0.0; D]),
                coordinates,
                elements: Vec::new(),
                levels: 0,
                min_length: Quantity::new(1.0),
                scale,
                targets: Vec::new(),
            };
        }
        let mut min_coord = [Quantity::<Length>::new(Scalar::INFINITY); D];
        let mut max_coord = [Quantity::<Length>::new(Scalar::NEG_INFINITY); D];
        for point in coordinates {
            for ax in 0..D {
                min_coord[ax] = min_coord[ax].min(point[ax]);
                max_coord[ax] = max_coord[ax].max(point[ax]);
            }
        }
        let max_extent = (0..D)
            .map(|ax| max_coord[ax] - min_coord[ax])
            .fold(Quantity::default(), Quantity::max);
        let min_sdf = sdf
            .iter()
            .copied()
            .filter(|&value| value > Quantity::default())
            .fold(Quantity::new(Scalar::INFINITY), Quantity::min);
        let elements: Vec<&[usize]> = tessellation
            .mesh()
            .connectivities()
            .iter()
            .flatten()
            .collect();
        let triangles: Vec<[usize; 3]> = elements
            .iter()
            .map(|element| from_fn(|i| element[i]))
            .collect();
        let curvature = match tolerance {
            Some(tolerance) => sizing_field(
                &triangles,
                coordinates,
                tolerance,
                max_extent * floor_fraction,
                max_extent,
                gradation,
                Unresolved::Radius,
                Creases::Discarded,
            ),
            None => vec![max_extent; coordinates.len()],
        };
        let min_curvature = curvature
            .iter()
            .copied()
            .fold(Quantity::new(Scalar::INFINITY), Quantity::min);
        let thickness_length = if min_sdf.value().is_finite() {
            min_sdf / scale
        } else {
            max_extent
        };
        let min_length = thickness_length.min(min_curvature);
        let zero = Quantity::default();
        let levels = if max_extent <= zero || min_length <= zero {
            0u32
        } else {
            (max_extent / min_length + 2.0 * padding as Scalar)
                .log2()
                .ceil()
                .max(Quantity::default())
                .value() as u32
        };
        let targets: Vec<Quantity<Length>> = elements
            .iter()
            .map(|element| {
                let thickness = sdf[element[0]].min(sdf[element[1]]).min(sdf[element[2]]);
                let feature = curvature[element[0]]
                    .min(curvature[element[1]])
                    .min(curvature[element[2]])
                    * scale;
                thickness.min(feature)
            })
            .collect();
        Self {
            center: Coordinate::<D>::from(from_fn::<_, D, _>(|ax| {
                (min_coord[ax] + max_coord[ax]) / 2.0
            })),
            coordinates,
            elements,
            levels,
            min_length,
            scale,
            targets,
        }
    }
}
