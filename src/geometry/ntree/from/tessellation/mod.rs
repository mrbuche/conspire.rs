#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, CoordinateList,
        bbox::BoundingBox,
        mesh::Tessellation,
        ntree::{
            Octree,
            balance::Balancing,
            node::{Kind, Node, cell::Cell, slot::Slot},
            pair::Pairing,
            rescale::Rescaling,
            sizing::{Sizing, curvature::CurvatureSizing},
        },
    },
    math::{Quantity, Scalar},
};
use std::array::from_fn;

const D: usize = 3;
const M: usize = 6;

impl<T, U> Octree<T, U>
where
    T: Cell,
    U: Slot,
{
    /// Builds an octree from a tessellation, refining cells where either the
    /// local thickness or the local curvature demands a smaller size.
    ///
    /// `scale` controls cells-per-thickness; `curvature` controls
    /// curvature-driven refinement independent of thickness (e.g. a sphere
    /// has ~constant thickness everywhere but can still demand refinement
    /// from curvature alone). `padding` adds extra empty root levels in case
    /// the tessellation's boundary overlaps the primordial primal node.
    pub fn from_features(
        tessellation: &Tessellation,
        scale: Scalar,
        curvature: CurvatureSizing,
        padding: u16,
    ) -> Result<Self, &'static str> {
        Self::refine(&Sizing::new(tessellation, scale, curvature, padding))
    }
    /// Refines an octree to a given size field.
    pub fn refine(sizing: &Sizing) -> Result<Self, &'static str> {
        let Sizing {
            center,
            coordinates,
            elements,
            levels,
            min_length,
            scale,
            targets,
        } = sizing;
        let (center, min_length, scale) = (center, *min_length, *scale);
        if elements.is_empty() {
            return Ok(Self {
                balanced: Balancing::None,
                nodes: vec![Node {
                    corner: from_fn(|_| T::ZERO),
                    length: T::ONE,
                    facets: [None; M],
                    kind: Kind::Leaf,
                    value: None,
                }],
                paired: Pairing::None,
                rescale: Rescaling {
                    center: Coordinate::const_from([0.0; D]),
                    cell: Quantity::new(1.0),
                    half: 0.0,
                },
            });
        }
        let root_length = 1usize
            .checked_shl(*levels)
            .and_then(T::length)
            .ok_or("sizing field exceeds maximum octree depth")?;
        let half = root_length.scalar() / 2.0;
        let mut tree = Self {
            balanced: Balancing::None,
            rescale: Rescaling {
                center: center.clone(),
                cell: min_length,
                half,
            },
            nodes: vec![Node {
                corner: from_fn(|_| T::ZERO),
                length: root_length,
                facets: [None; M],
                kind: Kind::Leaf,
                value: None,
            }],
            paired: Pairing::None,
        };
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
            let cells: usize = tree.nodes[index].length.cells();
            let extent: Scalar = tree.nodes[index].length.scalar();
            let target = overlapping
                .iter()
                .map(|&triangle| targets[triangle])
                .fold(Quantity::new(Scalar::INFINITY), Quantity::min);
            if min_length * (extent * scale) <= target {
                continue;
            }
            if cells <= 1 {
                continue;
            }
            tree.subdivide(index)?;
            let children: Vec<usize> = tree.nodes[index]
                .orthants()
                .unwrap()
                .iter()
                .map(|&child| child.slot())
                .collect();
            for child in children {
                let corner = tree.nodes[child].corner;
                let child_extent: Scalar = tree.nodes[child].length.scalar();
                let minimum = Coordinate::<3>::from(from_fn::<_, 3, _>(|ax| {
                    center[ax] + min_length * (corner[ax].scalar() - half)
                }));
                let maximum = Coordinate::<3>::from(from_fn::<_, 3, _>(|ax| {
                    minimum[ax] + min_length * child_extent
                }));
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
        Ok(tree)
    }
}
