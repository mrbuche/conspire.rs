use crate::geometry::ntree::node::slot::Slot;
#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh},
        ntree::{
            Quadtree,
            dual::{Dualization, Initialize, NodeMap, Star},
            node::cell::Cell,
        },
    },
    math::{Scalar, TensorVec},
};

const D: usize = 2;
const N: usize = 4;

impl<T, U> Dualization<D> for Quadtree<T, U>
where
    T: Cell,
    U: Slot,
{
    fn dualize(&mut self) -> Mesh<D> {
        let (center_nodes, mut coordinates, mut node_index, mut connectivity) = self.initialize();
        let mut nodes_map = NodeMap::new();
        edge_transition(
            self,
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

fn edge_transition<T, U>(
    tree: &Quadtree<T, U>,
    center_nodes: &[usize],
    coordinates: &mut Coordinates<D>,
    connectivity: &mut Vec<[usize; N]>,
    node_index: &mut usize,
    nodes_map: &mut NodeMap<D>,
) where
    T: Cell,
    U: Slot,
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
            let face_leaves = tree.orthants_leaves_on_facet(&tree[neighbor], 1);
            if let Some([Some(g_0a), Some(g_0b)]) = face_leaves[0]
                && let Some([Some(g_2a), Some(g_2b)]) = face_leaves[1]
            {
                let length: Scalar = tree[g_2a].length.scalar();
                let x0: Scalar = tree[g_2a].corner[0].scalar();
                let y0c: Scalar = tree[g_2a].corner[1].scalar();
                let x1 = x0 + length;
                let y0 = y0c - length * 0.5;
                let y1 = y0 + length;
                let new_1 = get_or_add([x1, y0]);
                let new_2 = get_or_add([x1, y1]);
                connectivity.push([
                    center_nodes[g_0b.slot()],
                    new_1,
                    new_2,
                    center_nodes[g_2a.slot()],
                ]);
                connectivity.push([
                    new_1,
                    center_nodes[leaf_0.slot()],
                    center_nodes[leaf_2.slot()],
                    new_2,
                ]);
                connectivity.push([
                    center_nodes[g_2a.slot()],
                    new_2,
                    center_nodes[leaf_2.slot()],
                    center_nodes[g_2b.slot()],
                ]);
                connectivity.push([
                    center_nodes[g_0a.slot()],
                    center_nodes[leaf_0.slot()],
                    new_1,
                    center_nodes[g_0b.slot()],
                ]);
            }
        }
        if let Some(neighbor) = node.facets()[1]
            && let Some(leaf_1) = node_leaves[1]
            && let Some(leaf_3) = node_leaves[3]
        {
            let face_leaves = tree.orthants_leaves_on_facet(&tree[neighbor], 0);
            if let Some([Some(g_1a), Some(g_1b)]) = face_leaves[0]
                && let Some([Some(g_3a), Some(g_3b)]) = face_leaves[1]
            {
                let length: Scalar = tree[g_3a].length.scalar();
                let x0: Scalar = tree[g_3a].corner[0].scalar();
                let y0c: Scalar = tree[g_3a].corner[1].scalar();
                let x1 = x0;
                let y0 = y0c - length * 0.5;
                let y1 = y0 + length;
                let new_1 = get_or_add([x1, y0]);
                let new_2 = get_or_add([x1, y1]);
                connectivity.push([
                    new_1,
                    center_nodes[g_1b.slot()],
                    center_nodes[g_3a.slot()],
                    new_2,
                ]);
                connectivity.push([
                    new_1,
                    new_2,
                    center_nodes[leaf_3.slot()],
                    center_nodes[leaf_1.slot()],
                ]);
                connectivity.push([
                    center_nodes[g_3a.slot()],
                    center_nodes[g_3b.slot()],
                    center_nodes[leaf_3.slot()],
                    new_2,
                ]);
                connectivity.push([
                    center_nodes[g_1a.slot()],
                    center_nodes[g_1b.slot()],
                    new_1,
                    center_nodes[leaf_1.slot()],
                ]);
            }
        }
        if let Some(neighbor) = node.facets()[2]
            && let Some(leaf_0) = node_leaves[0]
            && let Some(leaf_1) = node_leaves[1]
        {
            let face_leaves = tree.orthants_leaves_on_facet(&tree[neighbor], 3);
            if let Some([Some(g_0a), Some(g_0b)]) = face_leaves[0]
                && let Some([Some(g_1a), Some(g_1b)]) = face_leaves[1]
            {
                let length: Scalar = tree[g_1a].length.scalar();
                let x0c: Scalar = tree[g_1a].corner[0].scalar();
                let y0: Scalar = tree[g_1a].corner[1].scalar();
                let y1 = y0 + length;
                let x0 = x0c - length * 0.5;
                let x1 = x0 + length;
                let new_1 = get_or_add([x0, y1]);
                let new_2 = get_or_add([x1, y1]);
                connectivity.push([
                    center_nodes[g_0b.slot()],
                    center_nodes[g_1a.slot()],
                    new_2,
                    new_1,
                ]);
                connectivity.push([
                    new_1,
                    new_2,
                    center_nodes[leaf_1.slot()],
                    center_nodes[leaf_0.slot()],
                ]);
                connectivity.push([
                    center_nodes[g_1a.slot()],
                    center_nodes[g_1b.slot()],
                    center_nodes[leaf_1.slot()],
                    new_2,
                ]);
                connectivity.push([
                    center_nodes[g_0a.slot()],
                    center_nodes[g_0b.slot()],
                    new_1,
                    center_nodes[leaf_0.slot()],
                ]);
            }
        }
        if let Some(neighbor) = node.facets()[3]
            && let Some(leaf_2) = node_leaves[2]
            && let Some(leaf_3) = node_leaves[3]
        {
            let face_leaves = tree.orthants_leaves_on_facet(&tree[neighbor], 2);
            if let Some([Some(g_2a), Some(g_2b)]) = face_leaves[0]
                && let Some([Some(g_3a), Some(g_3b)]) = face_leaves[1]
            {
                let length: Scalar = tree[g_3a].length.scalar();
                let x0c: Scalar = tree[g_3a].corner[0].scalar();
                let y0c: Scalar = tree[g_3a].corner[1].scalar();
                let y1 = y0c;
                let x0 = x0c - length * 0.5;
                let x1 = x0 + length;
                let new_1 = get_or_add([x0, y1]);
                let new_2 = get_or_add([x1, y1]);
                connectivity.push([
                    new_1,
                    new_2,
                    center_nodes[g_3a.slot()],
                    center_nodes[g_2b.slot()],
                ]);
                connectivity.push([
                    new_1,
                    center_nodes[leaf_2.slot()],
                    center_nodes[leaf_3.slot()],
                    new_2,
                ]);
                connectivity.push([
                    center_nodes[g_3a.slot()],
                    new_2,
                    center_nodes[leaf_3.slot()],
                    center_nodes[g_3b.slot()],
                ]);
                connectivity.push([
                    center_nodes[g_2a.slot()],
                    center_nodes[leaf_2.slot()],
                    new_1,
                    center_nodes[g_2b.slot()],
                ]);
            }
        }
    });
}
