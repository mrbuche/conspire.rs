use crate::{
    geometry::{
        Coordinate, CoordinateList,
        bbox::BoundingBox,
        mesh::Tessellation,
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

impl<T, U> Octree<T, U>
where
    T: Add<Output = T> + Copy + From<u16> + Into<Scalar> + Into<usize> + PartialOrd + Split,
    U: Copy + From<usize> + Into<usize>,
{
    pub fn from_sdf(tessellation: &Tessellation, scale: Scalar) -> Self {
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
                corner: from_fn(|_| T::from(0)),
                length: T::from(root_length),
                facets: [None; M],
                kind: Kind::Leaf,
            }],
            paired: Pairing::None,
        };
        let elements: Vec<&[usize]> = tessellation
            .mesh()
            .connectivities()
            .iter()
            .flatten()
            .collect();
        let targets: Vec<Scalar> = elements
            .iter()
            .map(|element| sdf[element[0]].min(sdf[element[1]]).min(sdf[element[2]]))
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
