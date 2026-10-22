#[cfg(test)]
mod test;

pub mod curvature;
pub mod separation;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{
            Along, Tessellation,
            differential::sizing::{Creases, Unresolved, sizing_field},
        },
        ntree::{
            node::cell::Cell,
            sizing::{
                curvature::CurvatureSizing,
                separation::{SeparationSizing, asks},
            },
        },
    },
    math::{Quantity, Scalar, Tensor, TensorVec},
    units::Length,
};
use std::{array::from_fn, f64::consts::FRAC_PI_4};

const D: usize = 3;

/// Two creases, how close they come, and how far apart they are along the
/// creases.
type Narrow = ([[Coordinate<D>; 2]; 2], Quantity<Length>, Along);

/// Parameters controlling octree refinement.
pub struct Sizing<'a> {
    pub(crate) center: Coordinate<D>,
    pub(crate) coordinates: &'a Coordinates<D>,
    pub(crate) elements: Vec<&'a [usize]>,
    pub(crate) levels: u32,
    pub(crate) min_length: Quantity<Length>,
    pub(crate) along: Vec<Along>,
    pub(crate) narrow: Vec<[[Coordinate<D>; 2]; 2]>,
    pub(crate) ratio: Scalar,
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
        separation: SeparationSizing,
        padding: u16,
    ) -> Self {
        let CurvatureSizing {
            tolerance,
            gradation,
            floor_fraction,
        } = curvature;
        let SeparationSizing {
            minimum: separation_minimum,
            scale: separation_scale,
        } = separation;
        let separation_scale = separation_scale.unwrap_or(scale);
        let sdf = tessellation.shape_diameter_function(FRAC_PI_4, 3, 10);
        let coordinates = tessellation.mesh().coordinates();
        if coordinates.is_empty() {
            return Self {
                center: Coordinate::const_from([0.0; D]),
                coordinates,
                elements: Vec::new(),
                levels: 0,
                min_length: Quantity::new(1.0),
                along: Vec::new(),
                narrow: Vec::new(),
                ratio: 1.0,
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
                Creases::Excluded,
            ),
            None => vec![max_extent; coordinates.len()],
        };
        let min_curvature = curvature
            .iter()
            .copied()
            .fold(Quantity::new(Scalar::INFINITY), Quantity::min);
        // Narrow features are kept as the pairs of creases they were measured
        // between, and applied below only to cells lying between a pair,
        // rather than smeared onto a per-vertex size field. A crease bounding
        // a narrow feature is typically also a vertex of the large, flat
        // triangles covering the bulk of the surface, and is often long while
        // only part of it runs close to anything, so a field carried on either
        // the triangles or the whole creases would drive the entire model down
        // to the gap size instead of just the gap. Every pair a crease takes
        // part in is kept, since each covers a different stretch of it.
        let narrow_pairs: Vec<Narrow> = match separation_minimum {
            Some(minimum) => {
                let features = tessellation.features();
                let apart = features
                    .separation(tessellation, minimum)
                    .into_iter()
                    .enumerate()
                    .flat_map(|(crease, partners)| {
                        // Each pair is reported from both of its creases, and
                        // one copy of it is enough. Creases that touch have
                        // no gap to resolve.
                        partners
                            .into_iter()
                            .filter(move |partner| {
                                crease < partner.crease && partner.distance > Quantity::default()
                            })
                            .map(move |partner| {
                                (
                                    [
                                        features.creases()[crease].clone(),
                                        features.creases()[partner.crease].clone(),
                                    ],
                                    partner.distance,
                                    Along::Apart,
                                )
                            })
                    });
                // Creases of one chain, or of chains that meet, are narrow only
                // where they run close for how far apart they are along the
                // creases, and may reach down to a gap of nothing where they
                // meet, so they are taken only with a minimum to stop at, which
                // is the gap they are then sized to.
                let bends = if minimum > Quantity::default() {
                    features.bends(tessellation, minimum, asks)
                } else {
                    Vec::new()
                };
                apart
                    .chain(bends.into_iter().map(|bend| {
                        (
                            [
                                features.creases()[bend.creases[0]].clone(),
                                features.creases()[bend.creases[1]].clone(),
                            ],
                            minimum,
                            bend.along,
                        )
                    }))
                    .collect()
            }
            None => Vec::new(),
        };
        let min_narrow = narrow_pairs
            .iter()
            .map(|&(_, distance, _)| distance)
            .fold(Quantity::new(Scalar::INFINITY), Quantity::min);
        let thickness_length = if min_sdf.value().is_finite() {
            min_sdf / scale
        } else {
            max_extent
        };
        let narrow_length = if min_narrow.value().is_finite() {
            min_narrow / separation_scale
        } else {
            max_extent
        };
        let min_length = thickness_length.min(min_curvature).min(narrow_length);
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
        let (narrow, along): (Vec<[[Coordinate<D>; 2]; 2]>, Vec<Along>) = narrow_pairs
            .into_iter()
            .map(|(pair, _, along)| (pair, along))
            .unzip();
        Self {
            center: Coordinate::<D>::from(from_fn::<_, D, _>(|ax| {
                (min_coord[ax] + max_coord[ax]) / 2.0
            })),
            coordinates,
            elements,
            levels,
            min_length,
            along,
            narrow,
            ratio: scale / separation_scale,
            scale,
            targets,
        }
    }
}
