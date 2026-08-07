#[cfg(test)]
pub(crate) mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh},
        ntree::{
            Quadtree,
            dual::{Dualization, Initialize, NodeMap, Star, get_or_add, leaf_containing},
            node::split::Split,
        },
    },
    math::Scalar,
};
use std::{array::from_fn, ops::Add};

const D: usize = 2;
const N: usize = 4;

impl<T, U> Dualization<D> for Quadtree<T, U>
where
    T: Add<Output = T> + Copy + Into<Scalar> + Into<usize> + PartialOrd + Split,
    U: Copy + Into<usize>,
{
    fn dualize(&mut self) -> Mesh<D> {
        let (center_nodes, mut coordinates, mut node_index, mut connectivity) = self.initialize();
        let mut nodes_map = NodeMap::new();
        self.transitions(
            &center_nodes,
            &mut coordinates,
            &mut connectivity,
            &mut node_index,
            &mut nodes_map,
        );
        self.star(&center_nodes, &mut connectivity);
        self.rescale_coordinates(&mut coordinates);
        (
            vec![Connectivity::Quadrilateral(connectivity.into())],
            coordinates,
        )
            .into()
    }
}

impl<T, U> Quadtree<T, U>
where
    T: Add<Output = T> + Copy + Into<Scalar> + Into<usize> + PartialOrd + Split,
    U: Copy + Into<usize>,
{
    /// Fills the strip facing each facet of each paired block, where four of the block's fine
    /// cells meet two coarse leaves. Two Steiner points on the interface split that strip into
    /// four quads. Where the block hangs off the domain only one half of the strip is real, and
    /// the template degenerates to the single quad left once both Steiner points collapse onto
    /// the facet midpoint.
    fn transitions(
        &self,
        center_nodes: &[usize],
        coordinates: &mut Coordinates<D>,
        connectivity: &mut Vec<[usize; N]>,
        node_index: &mut usize,
        nodes_map: &mut NodeMap<D>,
    ) {
        let root = &self.nodes[0];
        let low: [i64; D] = from_fn(|axis| Into::<usize>::into(root.corner[axis]) as i64);
        let high: [i64; D] = from_fn(|axis| low[axis] + Into::<usize>::into(root.length) as i64);
        let cell_at = |corner: [i64; D], length: i64| -> Option<usize> {
            if (0..D).any(|axis| corner[axis] < low[axis] || corner[axis] + length > high[axis]) {
                return None;
            }
            let point = from_fn(|axis| corner[axis] as usize);
            let index = leaf_containing(self, &point);
            let node = &self.nodes[index];
            (length as usize == node.length.into()
                && (0..D).all(|axis| point[axis] == node.corner[axis].into()))
            .then_some(index)
        };
        let mut blocks: Vec<([usize; D], usize)> = self.pairing_vertices.iter().copied().collect();
        blocks.sort_unstable();
        for (block, length) in blocks {
            let center: [i64; D] = from_fn(|axis| block[axis] as i64);
            let (coarse, fine) = (length as i64, length as i64 / 2);
            for facet in 0..2 * D {
                let (axis, side) = (facet >> 1, facet & 1);
                let tangent = 1 - axis;
                let interface = center[axis] + if side == 1 { coarse } else { -coarse };
                let outside = if side == 1 {
                    interface
                } else {
                    interface - coarse
                };
                let inside = if side == 1 {
                    interface - fine
                } else {
                    interface
                };
                let base = center[tangent] - coarse;
                let at = |along, across| {
                    let mut corner = [0; D];
                    corner[axis] = along;
                    corner[tangent] = across;
                    corner
                };
                let coarse_cells: [Option<usize>; 2] =
                    from_fn(|j| cell_at(at(outside, base + j as i64 * coarse), coarse));
                let fine_cells: [Option<usize>; N] =
                    from_fn(|j| cell_at(at(inside, base + j as i64 * fine), fine));
                // Local frame: outward normal along `axis`, tangential sense chosen so the pair
                // is right-handed, which keeps every template quad wound the same way.
                let reversed = side == axis;
                let coarse_of = |k: usize| coarse_cells[if reversed { 1 - k } else { k }];
                let fine_of = |k: usize| fine_cells[if reversed { 3 - k } else { k }];
                let node_of = |cell: Option<usize>| cell.map(|index| center_nodes[index]);
                // A half is one coarse leaf with the two fine cells behind it. It is taken only
                // when whole, and the facet only when every absent half is provably outside the
                // domain: absent for any other reason means the transition belongs to another
                // block, and drawing here would double-cover it.
                let half_present = |k: usize| {
                    coarse_of(k).is_some()
                        && fine_of(2 * k).is_some()
                        && fine_of(2 * k + 1).is_some()
                };
                let half_outside = |k: usize| {
                    let j = if reversed { 1 - k } else { k } as i64;
                    base + j * coarse < low[tangent] || base + (j + 1) * coarse > high[tangent]
                };
                if !(0..2).any(half_present) || (0..2).any(|k| !half_present(k) && !half_outside(k))
                {
                    continue;
                }
                // Each Steiner point sits on the interface, opposite one of the two inner fine
                // cells. Pairing them that way is what lets a truncated facet fall out of the
                // same template: every quad below pairs each of its cells with a point at the
                // same position, so dropping the quads whose cells are missing leaves exactly
                // the truncated template, with no separate case to write.
                let mut steiner = |k: usize| {
                    fine_of(k).map(|_| {
                        let offset = (k as Scalar - 1.5) * fine as Scalar;
                        let mut point = [0.0; D];
                        point[axis] = interface as Scalar;
                        point[tangent] =
                            center[tangent] as Scalar + if reversed { -offset } else { offset };
                        get_or_add(point.into(), coordinates, nodes_map, node_index)
                    })
                };
                let (lower, upper) = (steiner(1), steiner(2));
                let (fine_of, coarse_of) = (|k| node_of(fine_of(k)), |k| node_of(coarse_of(k)));
                for quad in [
                    [fine_of(1), lower, upper, fine_of(2)],
                    [lower, coarse_of(0), coarse_of(1), upper],
                    [fine_of(2), upper, coarse_of(1), fine_of(3)],
                    [fine_of(0), coarse_of(0), lower, fine_of(1)],
                ] {
                    if quad.iter().all(Option::is_some) {
                        connectivity.push(from_fn(|k| quad[k].unwrap()))
                    }
                }
            }
        }
    }
}
