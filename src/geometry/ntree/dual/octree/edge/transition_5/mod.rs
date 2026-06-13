use crate::{
    geometry::{
        Coordinates,
        ntree::{
            Octree,
            dual::{
                NodeMap,
                octree::{D, M, N},
            },
        },
    },
    math::Scalar,
};

fn edge_orthants(facet_m: usize, facet_n: usize, side_m: usize, side_n: usize) -> [usize; 2] {
    let axis_m = facet_m >> 1;
    let axis_n = facet_n >> 1;
    let axis_t = 3 - axis_m - axis_n;
    let base = (side_m << axis_m) | (side_n << axis_n);
    [base, base | (1 << axis_t)]
}

#[allow(clippy::too_many_arguments, unused_variables)]
pub fn template<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    coordinates: &mut Coordinates<D>,
    connectivity: &mut Vec<[usize; N]>,
    node_index: &mut usize,
    nodes_map: &mut NodeMap<D>,
) where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    let child = |n: usize, o: usize| tree.nodes[n].orthants().unwrap()[o].into();
    let is_leaf = |n: usize, o: usize| tree.nodes[child(n, o)].is_leaf();
    for (cell_a, node) in tree.iter().enumerate() {
        if !node.is_leaf() {
            continue;
        }
        for facet_m in 0..M {
            for facet_n in (facet_m + 1)..M {
                if facet_m >> 1 == facet_n >> 1 {
                    continue;
                }
                let (Some(neighbor_m), Some(neighbor_n)) =
                    (node.facets[facet_m], node.facets[facet_n])
                else {
                    continue;
                };
                let cell_b = neighbor_m.into();
                let cell_c = neighbor_n.into();
                if !tree.nodes[cell_b].is_tree() || !tree.nodes[cell_c].is_tree() {
                    continue;
                }
                let Some(neighbor_diag) = tree.nodes[cell_b].facets[facet_n] else {
                    continue;
                };
                let cell_d = neighbor_diag.into();
                if !tree.nodes[cell_d].is_tree() {
                    continue;
                }
                let orthants_b = edge_orthants(facet_m, facet_n, (facet_m ^ 1) & 1, facet_n & 1);
                let orthants_c = edge_orthants(facet_m, facet_n, facet_m & 1, (facet_n ^ 1) & 1);
                let orthants_d =
                    edge_orthants(facet_m, facet_n, (facet_m ^ 1) & 1, (facet_n ^ 1) & 1);
                if orthants_b.into_iter().all(|o| is_leaf(cell_b, o))
                    && orthants_c.into_iter().all(|o| is_leaf(cell_c, o))
                    && orthants_d.into_iter().all(|o| {
                        let edge_child = child(cell_d, o);
                        tree.nodes[edge_child].is_tree()
                            && orthants_d.into_iter().all(|p| tree.nodes[child(edge_child, p)].is_leaf())
                    })
                {
                    println!(
                        "transition_5 (0,1,2,1): edge facets ({facet_m},{facet_n}) \
                         A={cell_a} B={cell_b} C={cell_c} D={cell_d}"
                    );
                }
            }
        }
    }
}
