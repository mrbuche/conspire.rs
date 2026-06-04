mod transition_1;
mod transition_2;

use super::{D, L, M, N};
use crate::geometry::ntree::{Octree, node::Node};

type Template<T, U> =
    fn(&Octree<T, U>, &Node<D, M, N, T, U>, &[U; N], &[usize], [usize; 11]) -> Option<[usize; N]>;

pub fn vertex_transitions<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    connectivity: &mut Vec<[usize; N]>,
) where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    apply(
        tree,
        center_nodes,
        connectivity,
        &transition_1::DATA,
        transition_1::template,
    );
    apply(
        tree,
        center_nodes,
        connectivity,
        &transition_2::DATA,
        transition_2::template,
    );
}

fn apply<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    connectivity: &mut Vec<[usize; N]>,
    data: &[[usize; 11]],
    template: Template<T, U>,
) where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    for node in tree.iter() {
        if let Some(cell_subcells) = tree.all_leaves(node) {
            for &row in data {
                if let Some(hex) = template(tree, node, cell_subcells, center_nodes, row) {
                    connectivity.push(hex)
                }
            }
        }
    }
}

/// The leaf at flattened sub-subcell `idx` (`idx / L` outer, `idx % L` inner) on the
/// shared face of `neighbor` (its mirror face, `facet ^ 1`), or `None` if absent.
fn sub_subnode<T, U>(tree: &Octree<T, U>, neighbor: U, facet: usize, idx: usize) -> Option<U>
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    tree.orthants_leaves_on_facet(&tree.nodes[neighbor.into()], facet ^ 1)[idx / L]
        .and_then(|inner| inner[idx % L])
}
