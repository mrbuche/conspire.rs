#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh},
        ntree::{
            Quadtree,
            balance::Balancing,
            dual::{Dualization, NodeMap, Uniform},
            node::split::Split,
        },
    },
    math::{Scalar, TensorVec},
};
use std::ops::Add;

const D: usize = 2;
const N: usize = 4;

impl<T, U> Dualization<D> for Quadtree<T, U>
where
    T: Add<Output = T> + Copy + Into<Scalar> + Into<usize> + Split,
    U: Copy + Into<usize>,
{
    fn dualize(&mut self) -> Mesh<D> {
        let (center_nodes, mut coordinates, mut node_index, mut connectivity) = self.initialize();
        self.uniform_transitions(&center_nodes, &mut connectivity);
        let mut nodes_map = NodeMap::new();
        edge_transition(
            self,
            &center_nodes,
            &mut coordinates,
            &mut connectivity,
            &mut node_index,
            &mut nodes_map,
        );
        vertex_transition_1(self, &center_nodes, &mut connectivity);
        vertex_transition_2(self, &center_nodes, &mut connectivity);
        vertex_transition_3(self, &center_nodes, &mut connectivity);
        vertex_transition_4(self, &center_nodes, &mut connectivity);
        if matches!(self.balanced, Balancing::Weak) {
            vertex_transition_5(self, &center_nodes, &mut connectivity);
        }
        (
            vec![Connectivity::Quadrilateral(connectivity.into())],
            coordinates,
        )
            .into()
    }
}

fn edge_transition<T, U>(
    tree: &Quadtree<T, U>,
    center_nodes: &[usize],
    coordinates: &mut Coordinates<D>,
    connectivity: &mut Vec<[usize; N]>,
    node_index: &mut usize,
    nodes_map: &mut NodeMap<D>,
) where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    let mut get_or_add = |pos: [Scalar; D]| -> usize {
        let key = pos.map(|p| (2.0 * p) as usize);
        if let Some(&v) = nodes_map.get(&key) {
            v
        } else {
            let v = *node_index;
            coordinates.push(pos.into());
            nodes_map.insert(key, v);
            *node_index += 1;
            v
        }
    };
    tree.iter().for_each(|node| {
        let node_leaves = tree.leaves(node);
        if let Some(neighbor) = node.facets()[0]
            && let Some(leaf_0) = node_leaves[0]
            && let Some(leaf_2) = node_leaves[2]
        {
            let n_orthants = tree.orthants_leaves(&tree[neighbor]);
            if let Some(o1_kids) = n_orthants[1]
                && let Some(g_0a) = o1_kids[1]
                && let Some(g_0b) = o1_kids[3]
                && let Some(o3_kids) = n_orthants[3]
                && let Some(g_2a) = o3_kids[1]
                && let Some(g_2b) = o3_kids[3]
            {
                let length: Scalar = tree[g_2a].length.into();
                let x0: Scalar = tree[g_2a].corner[0].into();
                let y0c: Scalar = tree[g_2a].corner[1].into();
                let x1 = x0 + length;
                let y0 = y0c - length * 0.5;
                let y1 = y0 + length;
                let new_1 = get_or_add([x1, y0]);
                let new_2 = get_or_add([x1, y1]);
                connectivity.push([
                    center_nodes[g_0b.into()],
                    new_1,
                    new_2,
                    center_nodes[g_2a.into()],
                ]);
                connectivity.push([
                    new_1,
                    center_nodes[leaf_0.into()],
                    center_nodes[leaf_2.into()],
                    new_2,
                ]);
                connectivity.push([
                    center_nodes[g_2a.into()],
                    new_2,
                    center_nodes[leaf_2.into()],
                    center_nodes[g_2b.into()],
                ]);
                connectivity.push([
                    center_nodes[g_0a.into()],
                    center_nodes[leaf_0.into()],
                    new_1,
                    center_nodes[g_0b.into()],
                ]);
            }
        }
        if let Some(neighbor) = node.facets()[1]
            && let Some(leaf_1) = node_leaves[1]
            && let Some(leaf_3) = node_leaves[3]
        {
            let n_orthants = tree.orthants_leaves(&tree[neighbor]);
            if let Some(o0_kids) = n_orthants[0]
                && let Some(g_1a) = o0_kids[0]
                && let Some(g_1b) = o0_kids[2]
                && let Some(o2_kids) = n_orthants[2]
                && let Some(g_3a) = o2_kids[0]
                && let Some(g_3b) = o2_kids[2]
            {
                let length: Scalar = tree[g_3a].length.into();
                let x0: Scalar = tree[g_3a].corner[0].into();
                let y0c: Scalar = tree[g_3a].corner[1].into();
                let x1 = x0;
                let y0 = y0c - length * 0.5;
                let y1 = y0 + length;
                let new_1 = get_or_add([x1, y0]);
                let new_2 = get_or_add([x1, y1]);
                connectivity.push([
                    new_1,
                    center_nodes[g_1b.into()],
                    center_nodes[g_3a.into()],
                    new_2,
                ]);
                connectivity.push([
                    new_1,
                    new_2,
                    center_nodes[leaf_3.into()],
                    center_nodes[leaf_1.into()],
                ]);
                connectivity.push([
                    center_nodes[g_3a.into()],
                    center_nodes[g_3b.into()],
                    center_nodes[leaf_3.into()],
                    new_2,
                ]);
                connectivity.push([
                    center_nodes[g_1a.into()],
                    center_nodes[g_1b.into()],
                    new_1,
                    center_nodes[leaf_1.into()],
                ]);
            }
        }
        if let Some(neighbor) = node.facets()[2]
            && let Some(leaf_0) = node_leaves[0]
            && let Some(leaf_1) = node_leaves[1]
        {
            let n_orthants = tree.orthants_leaves(&tree[neighbor]);
            if let Some(o2_kids) = n_orthants[2]
                && let Some(g_0a) = o2_kids[2]
                && let Some(g_0b) = o2_kids[3]
                && let Some(o3_kids) = n_orthants[3]
                && let Some(g_1a) = o3_kids[2]
                && let Some(g_1b) = o3_kids[3]
            {
                let length: Scalar = tree[g_1a].length.into();
                let x0c: Scalar = tree[g_1a].corner[0].into();
                let y0: Scalar = tree[g_1a].corner[1].into();
                let y1 = y0 + length;
                let x0 = x0c - length * 0.5;
                let x1 = x0 + length;
                let new_1 = get_or_add([x0, y1]);
                let new_2 = get_or_add([x1, y1]);
                connectivity.push([
                    center_nodes[g_0b.into()],
                    center_nodes[g_1a.into()],
                    new_2,
                    new_1,
                ]);
                connectivity.push([
                    new_1,
                    new_2,
                    center_nodes[leaf_1.into()],
                    center_nodes[leaf_0.into()],
                ]);
                connectivity.push([
                    center_nodes[g_1a.into()],
                    center_nodes[g_1b.into()],
                    center_nodes[leaf_1.into()],
                    new_2,
                ]);
                connectivity.push([
                    center_nodes[g_0a.into()],
                    center_nodes[g_0b.into()],
                    new_1,
                    center_nodes[leaf_0.into()],
                ]);
            }
        }
        if let Some(neighbor) = node.facets()[3]
            && let Some(leaf_2) = node_leaves[2]
            && let Some(leaf_3) = node_leaves[3]
        {
            let n_orthants = tree.orthants_leaves(&tree[neighbor]);
            if let Some(o0_kids) = n_orthants[0]
                && let Some(g_2a) = o0_kids[0]
                && let Some(g_2b) = o0_kids[1]
                && let Some(o1_kids) = n_orthants[1]
                && let Some(g_3a) = o1_kids[0]
                && let Some(g_3b) = o1_kids[1]
            {
                let length: Scalar = tree[g_3a].length.into();
                let x0c: Scalar = tree[g_3a].corner[0].into();
                let y0c: Scalar = tree[g_3a].corner[1].into();
                let y1 = y0c;
                let x0 = x0c - length * 0.5;
                let x1 = x0 + length;
                let new_1 = get_or_add([x0, y1]);
                let new_2 = get_or_add([x1, y1]);
                connectivity.push([
                    new_1,
                    new_2,
                    center_nodes[g_3a.into()],
                    center_nodes[g_2b.into()],
                ]);
                connectivity.push([
                    new_1,
                    center_nodes[leaf_2.into()],
                    center_nodes[leaf_3.into()],
                    new_2,
                ]);
                connectivity.push([
                    center_nodes[g_3a.into()],
                    new_2,
                    center_nodes[leaf_3.into()],
                    center_nodes[g_3b.into()],
                ]);
                connectivity.push([
                    center_nodes[g_2a.into()],
                    center_nodes[leaf_2.into()],
                    new_1,
                    center_nodes[g_2b.into()],
                ]);
            }
        }
    });
}

fn vertex_transition_1<T, U>(
    tree: &Quadtree<T, U>,
    center_nodes: &[usize],
    connectivity: &mut Vec<[usize; N]>,
) where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    tree.iter().for_each(|node| {
        let [leaf_0, leaf_1, leaf_2, leaf_3] = tree.leaves(node);
        if let Some(curr_leaf) = leaf_0
            && let Some(left) = node.facets()[0]
            && let Some(left_leaf) = tree.leaves(&tree[left])[1]
            && let Some(below) = node.facets()[2]
            && let Some(below_leaf) = tree.leaves(&tree[below])[2]
            && let Some(diag) = tree[below].facets()[0]
            && let Some(diag_orth_0) = tree.orthants_leaves(&tree[diag])[3]
            && let Some(diag_corner_leaf) = diag_orth_0[3]
        {
            connectivity.push([
                center_nodes[curr_leaf.into()],
                center_nodes[left_leaf.into()],
                center_nodes[diag_corner_leaf.into()],
                center_nodes[below_leaf.into()],
            ]);
        }
        if let Some(curr_leaf) = leaf_1
            && let Some(right) = node.facets()[1]
            && let Some(right_leaf) = tree.leaves(&tree[right])[0]
            && let Some(below) = node.facets()[2]
            && let Some(below_leaf) = tree.leaves(&tree[below])[3]
            && let Some(diag) = tree[below].facets()[1]
            && let Some(diag_orth_2) = tree.orthants_leaves(&tree[diag])[2]
            && let Some(diag_corner_leaf) = diag_orth_2[2]
        {
            connectivity.push([
                center_nodes[curr_leaf.into()],
                center_nodes[below_leaf.into()],
                center_nodes[diag_corner_leaf.into()],
                center_nodes[right_leaf.into()],
            ]);
        }
        if let Some(curr_leaf) = leaf_2
            && let Some(left) = node.facets()[0]
            && let Some(left_leaf) = tree.leaves(&tree[left])[3]
            && let Some(above) = node.facets()[3]
            && let Some(above_leaf) = tree.leaves(&tree[above])[0]
            && let Some(diag) = tree[above].facets()[0]
            && let Some(diag_orth_1) = tree.orthants_leaves(&tree[diag])[1]
            && let Some(diag_corner_leaf) = diag_orth_1[1]
        {
            connectivity.push([
                center_nodes[curr_leaf.into()],
                center_nodes[above_leaf.into()],
                center_nodes[diag_corner_leaf.into()],
                center_nodes[left_leaf.into()],
            ]);
        }
        if let Some(curr_leaf) = leaf_3
            && let Some(right) = node.facets()[1]
            && let Some(right_leaf) = tree.leaves(&tree[right])[2]
            && let Some(above) = node.facets()[3]
            && let Some(above_leaf) = tree.leaves(&tree[above])[1]
            && let Some(diag) = tree[above].facets()[1]
            && let Some(diag_orth_0) = tree.orthants_leaves(&tree[diag])[0]
            && let Some(diag_corner_leaf) = diag_orth_0[0]
        {
            connectivity.push([
                center_nodes[curr_leaf.into()],
                center_nodes[right_leaf.into()],
                center_nodes[diag_corner_leaf.into()],
                center_nodes[above_leaf.into()],
            ]);
        }
    });
}

fn vertex_transition_2<T, U>(
    tree: &Quadtree<T, U>,
    center_nodes: &[usize],
    connectivity: &mut Vec<[usize; N]>,
) where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    tree.iter().for_each(|node| {
        let [leaf_0, leaf_1, leaf_2, leaf_3] = tree.leaves(node);
        if let Some(curr_leaf) = leaf_2
            && let Some(above) = node.facets()[3]
            && let Some(above_leaf) = tree.leaves(&tree[above])[0]
            && let Some(left) = node.facets()[0]
            && let Some(left_orth_3) = tree.orthants_leaves(&tree[left])[3]
            && let Some(left_corner_leaf) = left_orth_3[3]
            && let Some(diag) = tree[left].facets()[3]
            && let Some(diag_orth_1) = tree.orthants_leaves(&tree[diag])[1]
            && let Some(diag_corner_leaf) = diag_orth_1[1]
        {
            connectivity.push([
                center_nodes[diag_corner_leaf.into()],
                center_nodes[left_corner_leaf.into()],
                center_nodes[curr_leaf.into()],
                center_nodes[above_leaf.into()],
            ]);
        }
        if let Some(curr_leaf) = leaf_1
            && let Some(below) = node.facets()[2]
            && let Some(below_leaf) = tree.leaves(&tree[below])[3]
            && let Some(right) = node.facets()[1]
            && let Some(right_orth_0) = tree.orthants_leaves(&tree[right])[0]
            && let Some(right_corner_leaf) = right_orth_0[0]
            && let Some(diag) = tree[right].facets()[2]
            && let Some(diag_orth_2) = tree.orthants_leaves(&tree[diag])[2]
            && let Some(diag_corner_leaf) = diag_orth_2[2]
        {
            connectivity.push([
                center_nodes[below_leaf.into()],
                center_nodes[diag_corner_leaf.into()],
                center_nodes[right_corner_leaf.into()],
                center_nodes[curr_leaf.into()],
            ]);
        }
        if let Some(curr_leaf) = leaf_0
            && let Some(left) = node.facets()[0]
            && let Some(left_leaf) = tree.leaves(&tree[left])[1]
            && let Some(below) = node.facets()[2]
            && let Some(below_orth_2) = tree.orthants_leaves(&tree[below])[2]
            && let Some(below_corner_leaf) = below_orth_2[2]
            && let Some(diag) = tree[below].facets()[0]
            && let Some(diag_orth_3) = tree.orthants_leaves(&tree[diag])[3]
            && let Some(diag_corner_leaf) = diag_orth_3[3]
        {
            connectivity.push([
                center_nodes[curr_leaf.into()],
                center_nodes[left_leaf.into()],
                center_nodes[diag_corner_leaf.into()],
                center_nodes[below_corner_leaf.into()],
            ]);
        }
        if let Some(curr_leaf) = leaf_3
            && let Some(right) = node.facets()[1]
            && let Some(right_leaf) = tree.leaves(&tree[right])[2]
            && let Some(above) = node.facets()[3]
            && let Some(above_orth_1) = tree.orthants_leaves(&tree[above])[1]
            && let Some(above_corner_leaf) = above_orth_1[1]
            && let Some(diag) = tree[above].facets()[1]
            && let Some(diag_orth_0) = tree.orthants_leaves(&tree[diag])[0]
            && let Some(diag_corner_leaf) = diag_orth_0[0]
        {
            connectivity.push([
                center_nodes[curr_leaf.into()],
                center_nodes[right_leaf.into()],
                center_nodes[diag_corner_leaf.into()],
                center_nodes[above_corner_leaf.into()],
            ]);
        }
    });
}

fn vertex_transition_3<T, U>(
    tree: &Quadtree<T, U>,
    center_nodes: &[usize],
    connectivity: &mut Vec<[usize; N]>,
) where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    tree.iter().for_each(|node| {
        let [leaf_0, leaf_1, _, _] = tree.leaves(node);
        if let Some(curr_leaf) = leaf_0
            && let Some(left) = node.facets()[0]
            && let Some(left_orth) = tree.orthants_leaves(&tree[left])[1]
            && let Some(left_leaf) = left_orth[1]
            && let Some(below) = node.facets()[2]
            && let Some(below_orth) = tree.orthants_leaves(&tree[below])[2]
            && let Some(below_leaf) = below_orth[2]
            && let Some(diag) = tree[below].facets()[0]
            && let Some(diag_leaf) = tree.leaves(&tree[diag])[3]
        {
            connectivity.push([
                center_nodes[curr_leaf.into()],
                center_nodes[left_leaf.into()],
                center_nodes[diag_leaf.into()],
                center_nodes[below_leaf.into()],
            ]);
        }
        if let Some(curr_leaf) = leaf_1
            && let Some(right) = node.facets()[1]
            && let Some(right_orth) = tree.orthants_leaves(&tree[right])[0]
            && let Some(right_leaf) = right_orth[0]
            && let Some(below) = node.facets()[2]
            && let Some(below_orth) = tree.orthants_leaves(&tree[below])[3]
            && let Some(below_leaf) = below_orth[3]
            && let Some(diag) = tree[below].facets()[1]
            && let Some(diag_leaf) = tree.leaves(&tree[diag])[2]
        {
            connectivity.push([
                center_nodes[right_leaf.into()],
                center_nodes[curr_leaf.into()],
                center_nodes[below_leaf.into()],
                center_nodes[diag_leaf.into()],
            ]);
        }
    })
}

fn vertex_transition_4<T, U>(
    tree: &Quadtree<T, U>,
    center_nodes: &[usize],
    connectivity: &mut Vec<[usize; N]>,
) where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    tree.iter().for_each(|node| {
        let [leaf_0, leaf_1, leaf_2, leaf_3] = tree.leaves(node);
        if let Some(leaf) = leaf_0
            && let Some(left) = node.facets()[0]
            && let Some(left_orth) = tree.orthants_leaves(&tree[left])[1]
            && let Some(left_leaf) = left_orth[1]
            && let Some(below) = node.facets()[2]
            && let Some(below_orth) = tree.orthants_leaves(&tree[below])[2]
            && let Some(below_leaf) = below_orth[2]
            && let Some(diag) = tree[below].facets()[0]
            && let Some(diag_orth) = tree.orthants_leaves(&tree[diag])[3]
            && let Some(diag_leaf) = diag_orth[3]
        {
            connectivity.push([
                center_nodes[leaf.into()],
                center_nodes[left_leaf.into()],
                center_nodes[diag_leaf.into()],
                center_nodes[below_leaf.into()],
            ]);
        }
        if let Some(leaf) = leaf_1
            && let Some(right) = node.facets()[1]
            && let Some(right_orth) = tree.orthants_leaves(&tree[right])[0]
            && let Some(right_leaf) = right_orth[0]
            && let Some(below) = node.facets()[2]
            && let Some(below_orth) = tree.orthants_leaves(&tree[below])[3]
            && let Some(below_leaf) = below_orth[3]
            && let Some(diag) = tree[below].facets()[1]
            && let Some(diag_orth) = tree.orthants_leaves(&tree[diag])[2]
            && let Some(diag_leaf) = diag_orth[2]
        {
            connectivity.push([
                center_nodes[leaf.into()],
                center_nodes[below_leaf.into()],
                center_nodes[diag_leaf.into()],
                center_nodes[right_leaf.into()],
            ]);
        }
        if let Some(leaf) = leaf_2
            && let Some(left) = node.facets()[0]
            && let Some(left_orth) = tree.orthants_leaves(&tree[left])[3]
            && let Some(left_leaf) = left_orth[3]
            && let Some(above) = node.facets()[3]
            && let Some(above_orth) = tree.orthants_leaves(&tree[above])[0]
            && let Some(above_leaf) = above_orth[0]
            && let Some(diag) = tree[above].facets()[0]
            && let Some(diag_orth) = tree.orthants_leaves(&tree[diag])[1]
            && let Some(diag_leaf) = diag_orth[1]
        {
            connectivity.push([
                center_nodes[leaf.into()],
                center_nodes[above_leaf.into()],
                center_nodes[diag_leaf.into()],
                center_nodes[left_leaf.into()],
            ]);
        }
        if let Some(leaf) = leaf_3
            && let Some(right) = node.facets()[1]
            && let Some(right_orth) = tree.orthants_leaves(&tree[right])[2]
            && let Some(right_leaf) = right_orth[2]
            && let Some(above) = node.facets()[3]
            && let Some(above_orth) = tree.orthants_leaves(&tree[above])[1]
            && let Some(above_leaf) = above_orth[1]
            && let Some(diag) = tree[above].facets()[1]
            && let Some(diag_orth) = tree.orthants_leaves(&tree[diag])[0]
            && let Some(diag_leaf) = diag_orth[0]
        {
            connectivity.push([
                center_nodes[leaf.into()],
                center_nodes[right_leaf.into()],
                center_nodes[diag_leaf.into()],
                center_nodes[above_leaf.into()],
            ]);
        }
    })
}

fn vertex_transition_5<T, U>(
    tree: &Quadtree<T, U>,
    center_nodes: &[usize],
    connectivity: &mut Vec<[usize; N]>,
) where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    tree.iter()
        .enumerate()
        .filter_map(|(index, node)| {
            if node.is_leaf() {
                Some((index, node.facets()))
            } else {
                None
            }
        })
        .for_each(|(leaf, &[facet_0, facet_1, facet_2, facet_3])| {
            if let Some(left) = facet_0
                && let Some(left_leaf) = tree.leaves(&tree[left])[1]
                && let Some(below) = facet_2
                && let Some(below_leaf) = tree.leaves(&tree[below])[2]
                && let Some(diag) = tree[below].facets()[0]
                && let Some(diag_orth) = tree.orthants_leaves(&tree[diag])[3]
                && let Some(diag_leaf) = diag_orth[3]
            {
                connectivity.push([
                    center_nodes[leaf],
                    center_nodes[left_leaf.into()],
                    center_nodes[diag_leaf.into()],
                    center_nodes[below_leaf.into()],
                ]);
            }
            if let Some(right) = facet_1
                && let Some(right_leaf) = tree.leaves(&tree[right])[0]
                && let Some(below) = facet_2
                && let Some(below_leaf) = tree.leaves(&tree[below])[3]
                && let Some(diag) = tree[below].facets()[1]
                && let Some(diag_orth) = tree.orthants_leaves(&tree[diag])[2]
                && let Some(diag_leaf) = diag_orth[2]
            {
                connectivity.push([
                    center_nodes[leaf],
                    center_nodes[below_leaf.into()],
                    center_nodes[diag_leaf.into()],
                    center_nodes[right_leaf.into()],
                ]);
            }
            if let Some(left) = facet_0
                && let Some(left_leaf) = tree.leaves(&tree[left])[3]
                && let Some(above) = facet_3
                && let Some(above_leaf) = tree.leaves(&tree[above])[0]
                && let Some(diag) = tree[above].facets()[0]
                && let Some(diag_orth) = tree.orthants_leaves(&tree[diag])[1]
                && let Some(diag_leaf) = diag_orth[1]
            {
                connectivity.push([
                    center_nodes[leaf],
                    center_nodes[above_leaf.into()],
                    center_nodes[diag_leaf.into()],
                    center_nodes[left_leaf.into()],
                ]);
            }
            if let Some(right) = facet_1
                && let Some(right_leaf) = tree.leaves(&tree[right])[2]
                && let Some(above) = facet_3
                && let Some(above_leaf) = tree.leaves(&tree[above])[1]
                && let Some(diag) = tree[above].facets()[1]
                && let Some(diag_orth) = tree.orthants_leaves(&tree[diag])[0]
                && let Some(diag_leaf) = diag_orth[0]
            {
                connectivity.push([
                    center_nodes[leaf],
                    center_nodes[right_leaf.into()],
                    center_nodes[diag_leaf.into()],
                    center_nodes[above_leaf.into()],
                ]);
            }
        })
}
