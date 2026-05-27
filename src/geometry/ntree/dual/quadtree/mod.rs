#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinates, Dualization, QuadrilateralMesh, Quadtree, ntree::node::split::Split},
    math::{Scalar, TensorVec},
};
use std::ops::Add;

const D: usize = 2;
const M: usize = 2;
const N: usize = 4;

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
                coordinates.push(leaf.center().map(Into::into).into());
                node_index += 1;
            });
        let mut connectivity = Vec::with_capacity(num);
        base_template_1(self, &center_nodes, &mut connectivity);
        base_template_2(self, &center_nodes, &mut connectivity);
        base_template_3(self, &center_nodes, &mut connectivity);
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
    connectivity.extend(tree.iter().filter_map(|node| tree.leaves(node)).map(
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
    tree.iter().for_each(|cell| {
        if let Some(cell_subcells) = tree.leaves(cell) {
            if let Some(neighbor) = cell.facets()[0]
                && let Some(neighbor_subcells) = tree.leaves(&tree[neighbor])
            {
                connectivity.push([
                    center_nodes[neighbor_subcells[1].into()],
                    center_nodes[cell_subcells[0].into()],
                    center_nodes[cell_subcells[2].into()],
                    center_nodes[neighbor_subcells[3].into()],
                ]);
            }
            if let Some(neighbor) = cell.facets()[2]
                && let Some(neighbor_subcells) = tree.leaves(&tree[neighbor])
            {
                connectivity.push([
                    center_nodes[neighbor_subcells[2].into()],
                    center_nodes[neighbor_subcells[3].into()],
                    center_nodes[cell_subcells[1].into()],
                    center_nodes[cell_subcells[0].into()],
                ]);
            }
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
    tree.iter().for_each(|cell| {
        if let Some(cell_subcells) = tree.leaves(cell)
            && let Some(nx) = cell.facets()[0]
            && let Some(nx_subcells) = tree.leaves(&tree[nx])
            && let Some(ny) = cell.facets()[2]
            && let Some(ny_subcells) = tree.leaves(&tree[ny])
            && let Some(diag) = tree[nx].facets()[2]
            && let Some(diag_subcells) = tree.leaves(&tree[diag])
        {
            connectivity.push([
                center_nodes[diag_subcells[3].into()],
                center_nodes[ny_subcells[2].into()],
                center_nodes[cell_subcells[0].into()],
                center_nodes[nx_subcells[1].into()],
            ]);
        }
    });
}
