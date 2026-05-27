#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinates, Dualization, QuadrilateralMesh, Quadtree, ntree::node::split::Split},
    math::{Scalar, TensorVec},
};
use std::{array::from_fn, collections::HashMap, ops::Add};

const D: usize = 2;
const M: usize = 2;
const N: usize = 4;

type NodeMap<V> = HashMap<[usize; D], V>;

impl<const I: usize, T, U, V> Dualization<D, I, M, N, V> for Quadtree<T, U>
where
    T: Add<Output = T> + Copy + Into<Scalar> + Into<usize> + Split,
    U: Copy + Into<usize>,
    V: Copy + Default + From<usize>,
{
    fn dualize(&mut self) -> QuadrilateralMesh<D, I, V> {
        let num = self.len();
        let mut center_nodes = vec![V::default(); num];
        let mut coordinates = Coordinates::with_capacity(num);
        let mut node_index = 0;
        self.iter()
            .enumerate()
            .filter(|(_, node)| node.is_leaf())
            .for_each(|(index, leaf)| {
                center_nodes[index] = V::from(node_index);
                let length: Scalar = leaf.length.into();
                let center: [Scalar; D] = from_fn(|i| {
                    let c: Scalar = leaf.corner[i].into();
                    c + length * 0.5
                });
                coordinates.push(center.into());
                node_index += 1;
            });
        let mut connectivity = Vec::with_capacity(num);
        let mut nodes_map: NodeMap<V> = HashMap::new();
        base_template_1(self, &center_nodes, &mut connectivity);
        base_template_2(self, &center_nodes, &mut connectivity);
        base_template_3(self, &center_nodes, &mut connectivity);
        edge_template_1(
            self,
            &center_nodes,
            &mut coordinates,
            &mut connectivity,
            &mut node_index,
            &mut nodes_map,
        );
        (connectivity, coordinates).into()
    }
}

fn base_template_1<T, U, V>(
    tree: &Quadtree<T, U>,
    center_nodes: &[V],
    connectivity: &mut Vec<[V; N]>,
) where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
    V: Copy,
{
    connectivity.extend(tree.iter().filter_map(|node| tree.all_leaves(node)).map(
        |&[leaf_0, leaf_1, leaf_2, leaf_3]| {
            [
                center_nodes[leaf_0.into()],
                center_nodes[leaf_1.into()],
                center_nodes[leaf_3.into()],
                center_nodes[leaf_2.into()],
            ]
        },
    ))
}

fn base_template_2<T, U, V>(
    tree: &Quadtree<T, U>,
    center_nodes: &[V],
    connectivity: &mut Vec<[V; N]>,
) where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
    V: Copy,
{
    tree.iter().for_each(|node| {
        let node_leaves = tree.leaves_and_facets(node);
        if let Some((leaf_0, facets_0)) = node_leaves[0]
            && let Some((leaf_2, facets_2)) = node_leaves[2]
            && let Some(n_leaf_1) = facets_0[0]
            && tree[n_leaf_1].is_leaf()
            && let Some(n_leaf_3) = facets_2[0]
            && tree[n_leaf_3].is_leaf()
        {
            connectivity.push([
                center_nodes[n_leaf_1.into()],
                center_nodes[leaf_0.into()],
                center_nodes[leaf_2.into()],
                center_nodes[n_leaf_3.into()],
            ]);
        }
        if let Some((leaf_0, facets_0)) = node_leaves[0]
            && let Some((leaf_1, facets_1)) = node_leaves[1]
            && let Some(n_leaf_2) = facets_0[1]
            && tree[n_leaf_2].is_leaf()
            && let Some(n_leaf_3) = facets_1[1]
            && tree[n_leaf_3].is_leaf()
        {
            connectivity.push([
                center_nodes[n_leaf_2.into()],
                center_nodes[n_leaf_3.into()],
                center_nodes[leaf_1.into()],
                center_nodes[leaf_0.into()],
            ]);
        }
    });
}

fn base_template_3<T, U, V>(
    tree: &Quadtree<T, U>,
    center_nodes: &[V],
    connectivity: &mut Vec<[V; N]>,
) where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
    V: Copy,
{
    tree.iter().for_each(|node| {
        let node_leaves = tree.leaves_and_facets(node);
        if let Some((leaf_0, facets_0)) = node_leaves[0]
            && let Some(n_leaf_1) = facets_0[0]
            && tree[n_leaf_1].is_leaf()
            && let Some(n_leaf_2) = facets_0[1]
            && tree[n_leaf_2].is_leaf()
            && let Some(n_leaf_diag) = tree[n_leaf_1].facets()[2]
            && tree[n_leaf_diag].is_leaf()
        {
            connectivity.push([
                center_nodes[n_leaf_diag.into()],
                center_nodes[n_leaf_2.into()],
                center_nodes[leaf_0.into()],
                center_nodes[n_leaf_1.into()],
            ]);
        }
    });
}

fn edge_template_1<const I: usize, T, U, V>(
    tree: &Quadtree<T, U>,
    center_nodes: &[V],
    coordinates: &mut Coordinates<D, I>,
    connectivity: &mut Vec<[V; N]>,
    node_index: &mut usize,
    nodes_map: &mut NodeMap<V>,
) where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
    V: Copy + From<usize>,
{
    let mut get_or_add = |pos: [Scalar; D]| -> V {
        let key: [usize; D] = pos.map(|p| (2.0 * p) as usize);
        if let Some(&v) = nodes_map.get(&key) {
            v
        } else {
            let v = V::from(*node_index);
            coordinates.push(pos.into());
            nodes_map.insert(key, v);
            *node_index += 1;
            v
        }
    };
    tree.iter().for_each(|node| {
        let node_leaves = tree.leaves(node);
        for face in 0..M {
            // Derive per-face slot/sub-slot indices via bit logic.
            //   axis = face normal direction (0 = x, 1 = y)
            //   side = which side of node (0 = -, 1 = +)
            //   perp = the in-face axis
            let axis = face / 2;
            let side = face % 2;
            let perp = 1 - axis;
            let make_slot = |v_axis: usize, v_perp: usize| (v_axis << axis) | (v_perp << perp);
            let leaf_low_s = make_slot(side, 0);
            let leaf_high_s = make_slot(side, 1);
            let mirror_low_s = make_slot(1 - side, 0);
            let mirror_high_s = make_slot(1 - side, 1);
            let sub_low_i = make_slot(1 - side, 0);
            let sub_high_i = make_slot(1 - side, 1);
            if let Some(neighbor) = node.facets()[face]
                && let Some(leaf_low) = node_leaves[leaf_low_s]
                && let Some(leaf_high) = node_leaves[leaf_high_s]
            {
                let n_orthants = tree.orthants_leaves(&tree[neighbor]);
                if let Some(o_low) = n_orthants[mirror_low_s]
                    && let Some(g_outer_low) = o_low[sub_low_i]
                    && let Some(g_inner_low) = o_low[sub_high_i]
                    && let Some(o_high) = n_orthants[mirror_high_s]
                    && let Some(g_inner_high) = o_high[sub_low_i]
                    && let Some(g_outer_high) = o_high[sub_high_i]
                {
                    let length: Scalar = tree[g_inner_high].length.into();
                    let half = length * 0.5;
                    let face_line: Scalar = {
                        let c: Scalar = tree[g_inner_high].corner[axis].into();
                        c + (1 - side) as Scalar * length
                    };
                    let foo_perp: Scalar = {
                        let c: Scalar = tree[g_inner_low].corner[perp].into();
                        c + half
                    };
                    let bar_perp: Scalar = {
                        let c: Scalar = tree[g_inner_high].corner[perp].into();
                        c + half
                    };
                    let mut foo_pos = [0.0; D];
                    foo_pos[axis] = face_line;
                    foo_pos[perp] = foo_perp;
                    let mut bar_pos = [0.0; D];
                    bar_pos[axis] = face_line;
                    bar_pos[perp] = bar_perp;
                    let foo = get_or_add(foo_pos);
                    let bar = get_or_add(bar_pos);
                    let ll = center_nodes[leaf_low.into()];
                    let lh = center_nodes[leaf_high.into()];
                    let gol = center_nodes[g_outer_low.into()];
                    let gil = center_nodes[g_inner_low.into()];
                    let gih = center_nodes[g_inner_high.into()];
                    let goh = center_nodes[g_outer_high.into()];
                    // Natural CCW order (for axis XOR side == 0, i.e. faces 0, 3).
                    let mut q1 = [gil, foo, bar, gih];
                    let mut q2 = [foo, ll, lh, bar];
                    let mut q3 = [gih, bar, lh, goh];
                    let mut q4 = [gol, ll, foo, gil];
                    // For faces 1, 2 the geometry is mirrored, so reverse CCW.
                    if (axis ^ side) != 0 {
                        for q in [&mut q1, &mut q2, &mut q3, &mut q4] {
                            q.swap(1, 3);
                        }
                    }
                    connectivity.push(q1);
                    connectivity.push(q2);
                    connectivity.push(q3);
                    connectivity.push(q4);
                }
            }
        }
    });
}
