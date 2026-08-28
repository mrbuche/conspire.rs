//! Meshing B-reps.

#[cfg(test)]
mod test;

use super::{brep::Brep, sizing::FeatureSizing};
use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Class, Connectivity, Mesh},
        ntree::{
            Balance, Balancing, Dualization, Orthotree, Pairing, Rescaling,
            node::{Kind, Node, slot::Slot},
        },
    },
    math::{Quantity, Scalar, Tensor},
    units::Length,
};
use std::{array::from_fn, num::NonZeroU32};

const D: usize = 3;

type Cube = Orthotree<D, 4, 6, 8, u16, NonZeroU32>;

impl Brep {
    /// Refines an octree over this solid's bounding box until every leaf is no
    /// larger than `sizing` allows at its centre, then returns the leaves as a
    /// hexahedral mesh.
    ///
    /// No tessellation, no balancing, no cutting: the mesh is a graded block
    /// of cubes covering the bounding box, hanging nodes and all.
    ///
    /// `max_levels` (1..=15) caps the octree depth, so the finest cell is the
    /// padded bounding box over `2^max_levels`. `padding` grows the box by
    /// that fraction on each side.
    pub fn sizing_octree(
        &self,
        sizing: &FeatureSizing,
        max_levels: u32,
        padding: Scalar,
    ) -> Result<Mesh<D>, &'static str> {
        let tree = self.refine_octree(sizing, max_levels, padding)?;
        let rescale = Rescaling {
            center: tree.rescale().center.clone(),
            cell: tree.rescale().cell,
            half: tree.rescale().half,
        };
        let (connectivity, mut coordinates): (Vec<[usize; 8]>, Coordinates<D>) = tree.into();
        coordinates
            .iter_mut()
            .for_each(|coordinate| *coordinate = rescale.apply(coordinate));
        Ok((
            vec![Connectivity::Hexahedral(connectivity.into())],
            coordinates,
        )
            .into())
    }

    /// Refines an octree over this solid's padded bounding box until every leaf
    /// is no larger than `sizing` allows at its centre. `max_levels` (1..=15)
    /// caps the depth; `padding` grows the box by that fraction per side.
    fn refine_octree(
        &self,
        sizing: &FeatureSizing,
        max_levels: u32,
        padding: Scalar,
    ) -> Result<Cube, &'static str> {
        if self.vertices.is_empty() {
            return Err("brep has no vertices");
        }
        if !(1..=15).contains(&max_levels) {
            return Err("max_levels must be in 1..=15");
        }
        let root_cells = 1u16 << max_levels;
        let mut low = [Scalar::INFINITY; D];
        let mut high = [Scalar::NEG_INFINITY; D];
        for vertex in &self.vertices {
            for (axis, (lo, hi)) in low.iter_mut().zip(high.iter_mut()).enumerate() {
                *lo = lo.min(vertex[axis].value());
                *hi = hi.max(vertex[axis].value());
            }
        }
        let center: Coordinate<D> = from_fn(|axis| 0.5 * (low[axis] + high[axis])).into();
        let side = (0..D)
            .map(|axis| high[axis] - low[axis])
            .fold(0.0, Scalar::max)
            * (1.0 + padding.max(0.0));
        if side <= 0.0 {
            return Err("degenerate bounding box");
        }
        let cell = Quantity::<Length>::new(side / Scalar::from(root_cells));
        let half = Scalar::from(root_cells) / 2.0;

        let mut tree: Cube = Orthotree {
            balanced: Balancing::None,
            paired: Pairing::None,
            rescale: Rescaling {
                center: center.clone(),
                cell,
                half,
            },
            nodes: vec![Node {
                corner: [0; D],
                length: root_cells,
                facets: [None; 6],
                kind: Kind::Leaf,
                value: None,
            }],
        };
        let physical = |cells: [u16; D]| -> Coordinate<D> {
            from_fn(|axis| cell.value() * (Scalar::from(cells[axis]) - half) + center[axis].value())
                .into()
        };
        let mut stack = vec![0usize];
        while let Some(index) = stack.pop() {
            let length = tree.nodes[index].length;
            if length <= 1 {
                continue;
            }
            let extent = cell * Scalar::from(length);
            if extent <= sizing.at(&physical(tree.nodes[index].center())) {
                continue;
            }
            tree.subdivide(index)?;
            let children: Vec<usize> = tree.nodes[index]
                .orthants()
                .expect("subdivided node has orthants")
                .iter()
                .map(|slot| slot.slot())
                .collect();
            stack.extend(children);
        }
        Ok(tree)
    }

    /// Refines the octree from `sizing`, balances and dualizes it, and
    /// classifies the resulting conforming all-hex mesh against this solid.
    /// `balancing` must be `Strong(1)` or `Weak(1)`.
    pub fn dual_background(
        &self,
        sizing: &FeatureSizing,
        max_levels: u32,
        padding: Scalar,
        balancing: Balancing,
    ) -> Result<(Mesh<D>, Vec<Class>), &'static str> {
        let mut tree = self.refine_octree(sizing, max_levels, padding)?;
        tree.equilibrate(balancing, Pairing::Regular)?;
        let mesh = tree.dualize();
        let classes = self.classify(&mesh)?;
        Ok((mesh, classes))
    }

    /// Builds the [`dual_background`](Self::dual_background) and drops the
    /// `Outside` cells. The kept `Cut` cells straddle the surface and come back
    /// for a later fit or cut step, so this is a trimmed background, not a
    /// finished mesh.
    pub fn trim(
        &self,
        sizing: &FeatureSizing,
        max_levels: u32,
        padding: Scalar,
        balancing: Balancing,
    ) -> Result<(Mesh<D>, Vec<Class>), &'static str> {
        let (mut mesh, classes) = self.dual_background(sizing, max_levels, padding, balancing)?;
        let keep: Vec<bool> = classes
            .iter()
            .map(|&class| class != Class::Outside)
            .collect();
        mesh.keep_hexes(|index, _, _| keep[index])?;
        let classes = classes
            .into_iter()
            .zip(&keep)
            .filter_map(|(class, &keep)| keep.then_some(class))
            .collect();
        Ok((mesh, classes))
    }
}
