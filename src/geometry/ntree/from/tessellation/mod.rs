use crate::{
    geometry::{
        mesh::Tessellation,
        ntree::{
            Octree,
            balance::Balancing,
            node::{Kind, Node},
            pair::Pairing,
            rescale::Rescaling,
        },
    },
    math::{Scalar, Tensor, TensorVec},
};
use std::{array::from_fn, f64::consts::FRAC_PI_3};

const D: usize = 3;
const M: usize = 6;

impl Octree<u16, usize> {
    pub fn from_sdf(tessellation: &Tessellation, scale: Scalar) -> Self {
        let sdf = tessellation.shape_diameter_function(FRAC_PI_3, 3, 8);
        let coordinates = tessellation.mesh().coordinates();
        if coordinates.is_empty() {
            return Self {
                balanced: Balancing::None,
                nodes: vec![Node {
                    corner: [0u16; D],
                    length: 1,
                    facets: [None; M],
                    kind: Kind::Leaf,
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
        // The finest cell (integer length 1) is sized to resolve the smallest
        // shape diameter, so that subdividing while `length * scale > sdf` halts.
        let min_sdf = sdf
            .iter()
            .copied()
            .filter(|value| *value > 0.0)
            .fold(f64::INFINITY, f64::min);
        let min_length = if min_sdf.is_finite() {
            min_sdf / scale
        } else {
            max_extent
        };
        let levels = if max_extent <= 0.0 || min_length <= 0.0 {
            0u32
        } else {
            (max_extent / min_length).log2().ceil().max(0.0) as u32
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
                corner: [0u16; D],
                length: root_length,
                facets: [None; M],
                kind: Kind::Leaf,
            }],
            paired: Pairing::None,
        };
        let int_coords: Vec<[u16; D]> = coordinates
            .iter()
            .map(|point| {
                from_fn(|ax| {
                    let v = ((point[ax] - center[ax]) / min_length + root_length as f64 / 2.0)
                        .floor() as i64;
                    v.clamp(0, root_length as i64 - 1) as u16
                })
            })
            .collect();
        for (vertex, int_coord) in int_coords.iter().enumerate() {
            let diameter = sdf[vertex];
            loop {
                let index = super::find_leaf(&tree, int_coord);
                let length = tree.nodes[index].length;
                // Local grid size in real units is `length * min_length`; keep
                // subdividing until it is at least `scale` times below the SDF.
                if length <= 1 || (length as f64 * min_length) * scale <= diameter {
                    break;
                }
                tree.subdivide(index).ok();
            }
        }
        tree
    }
}
