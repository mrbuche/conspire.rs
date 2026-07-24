#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, CoordinateList,
        bbox::BoundingBox,
        mesh::{Tessellation, remesh::adaptive::sizing_field},
        ntree::{
            Octree,
            balance::Balancing,
            node::{Kind, Node, split::Split},
            pair::Pairing,
            rescale::Rescaling,
        },
    },
    math::{Scalar, Tensor, TensorVec},
};
use std::{array::from_fn, f64::consts::FRAC_PI_3, ops::Add};

const D: usize = 3;
const M: usize = 6;

/// Parameters controlling curvature-driven octree refinement.
///
/// `tolerance` is the Dunyach chord-error tolerance (an absolute length, like
/// adaptive surface remeshing's) and has no sensible default, since it is
/// tied to the physical scale of the model; `None` disables curvature-driven
/// refinement entirely (thickness alone still governs cell size). `gradation`
/// and `floor_fraction` are dimensionless and default to values that work
/// well in practice.
pub struct CurvatureSizing {
    /// Chord-error tolerance; smaller refines more aggressively near curvature.
    /// `None` disables curvature-driven refinement.
    pub tolerance: Option<Scalar>,
    /// Lipschitz grading rate applied to the curvature-driven size field.
    pub gradation: Scalar,
    /// Smallest curvature-driven cell size allowed, as a fraction of the
    /// tessellation's bounding box extent (a safety valve near singular
    /// curvature, e.g. polytope corners).
    pub floor_fraction: Scalar,
}

impl Default for CurvatureSizing {
    fn default() -> Self {
        Self {
            tolerance: None,
            gradation: 0.5,
            floor_fraction: 1.0e-3,
        }
    }
}

impl<T, U> Octree<T, U>
where
    T: Add<Output = T> + Copy + From<u16> + Into<Scalar> + Into<usize> + PartialOrd + Split,
    U: Copy + From<usize> + Into<usize>,
{
    /// Builds an octree from a tessellation, refining cells where either the
    /// local thickness (SDF) or the local curvature demands a smaller size.
    ///
    /// `scale` controls cells-per-thickness, as before; `curvature` controls
    /// curvature-driven refinement independent of thickness (e.g. a sphere
    /// has ~constant thickness everywhere but can still demand refinement
    /// from curvature alone). `padding` adds extra empty root levels in case
    /// the tessellation's boundary overlaps the primordial primal node.
    pub fn from_features(
        tessellation: &Tessellation,
        scale: Scalar,
        curvature: CurvatureSizing,
        padding: u16,
    ) -> Self {
        let CurvatureSizing {
            tolerance,
            gradation,
            floor_fraction,
        } = curvature;
        let sdf = tessellation.shape_diameter_function(FRAC_PI_3, 3, 8);
        let coordinates = tessellation.mesh().coordinates();
        if coordinates.is_empty() {
            return Self {
                balanced: Balancing::None,
                nodes: vec![Node {
                    corner: from_fn(|_| T::from(0)),
                    length: T::from(1),
                    facets: [None; M],
                    kind: Kind::Leaf,
                    value: None,
                }],
                paired: Pairing::None,
                rescale: Rescaling {
                    center: [0.0; D],
                    cell: 1.0,
                    half: 0.0,
                },
            };
        }
        let mut min_coord: [f64; D] = from_fn(|_| f64::INFINITY);
        let mut max_coord: [f64; D] = from_fn(|_| f64::NEG_INFINITY);
        for point in coordinates {
            for ax in 0..D {
                min_coord[ax] = min_coord[ax].min(point[ax]);
                max_coord[ax] = max_coord[ax].max(point[ax]);
            }
        }
        let max_extent = (0..D)
            .map(|ax| max_coord[ax] - min_coord[ax])
            .fold(0.0f64, f64::max);
        let min_sdf = sdf
            .iter()
            .copied()
            .filter(|value| *value > 0.0)
            .fold(f64::INFINITY, f64::min);
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
            ),
            None => vec![max_extent; coordinates.len()],
        };
        let min_curvature = curvature.iter().copied().fold(f64::INFINITY, f64::min);
        let thickness_length = if min_sdf.is_finite() {
            min_sdf / scale
        } else {
            max_extent
        };
        let min_length = thickness_length.min(min_curvature);
        let levels = if max_extent <= 0.0 || min_length <= 0.0 {
            0u32
        } else {
            (max_extent / min_length + 2.0 * padding as Scalar)
                .log2()
                .ceil()
                .max(0.0) as u32
        };
        let root_length: u16 = 1u16.checked_shl(levels).unwrap_or(u16::MAX);
        let center: [f64; D] = from_fn(|ax| (min_coord[ax] + max_coord[ax]) / 2.0);
        let mut tree = Self {
            balanced: Balancing::None,
            rescale: Rescaling {
                center,
                cell: min_length,
                half: root_length as Scalar / 2.0,
            },
            nodes: vec![Node {
                corner: from_fn(|_| T::from(0)),
                length: T::from(root_length),
                facets: [None; M],
                kind: Kind::Leaf,
                value: None,
            }],
            paired: Pairing::None,
        };
        let targets: Vec<Scalar> = elements
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
        let half = root_length as Scalar / 2.0;
        let overlaps = |bbox: &BoundingBox<3>, triangle: usize| {
            let element = elements[triangle];
            bbox.overlaps_triangle(
                &coordinates[element[0]],
                &coordinates[element[1]],
                &coordinates[element[2]],
            )
        };
        let mut stack: Vec<(usize, Vec<usize>)> = vec![(0, (0..elements.len()).collect())];
        while let Some((index, overlapping)) = stack.pop() {
            let cells: usize = tree.nodes[index].length.into();
            let extent: Scalar = tree.nodes[index].length.into();
            let target = overlapping
                .iter()
                .map(|&triangle| targets[triangle])
                .fold(f64::INFINITY, f64::min);
            if cells <= 1 || (extent * min_length) * scale <= target {
                continue;
            }
            if tree.subdivide(U::from(index)).is_err() {
                continue;
            }
            let children: Vec<usize> = tree.nodes[index]
                .orthants()
                .unwrap()
                .iter()
                .map(|&child| child.into())
                .collect();
            for child in children {
                let corner = tree.nodes[child].corner;
                let child_extent: Scalar = tree.nodes[child].length.into();
                let minimum = Coordinate::const_from(from_fn(|ax| {
                    center[ax] + (Into::<Scalar>::into(corner[ax]) - half) * min_length
                }));
                let maximum =
                    Coordinate::const_from(from_fn(|ax| minimum[ax] + child_extent * min_length));
                let bbox = BoundingBox::from(CoordinateList::const_from([minimum, maximum]));
                let inside: Vec<usize> = overlapping
                    .iter()
                    .copied()
                    .filter(|&triangle| overlaps(&bbox, triangle))
                    .collect();
                if !inside.is_empty() {
                    stack.push((child, inside));
                }
            }
        }
        tree
    }
}
