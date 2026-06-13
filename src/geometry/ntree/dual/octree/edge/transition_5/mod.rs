use crate::{
    geometry::{
        Coordinate, Coordinates,
        ntree::{
            Octree,
            dual::{
                NodeMap, get_or_add,
                octree::{D, M, N, facet_direction},
            },
        },
    },
    math::Scalar,
};

#[allow(dead_code)] // fields consumed once connectivity is implemented
pub(crate) struct EdgeMatch {
    pub facet_m: usize,
    pub facet_n: usize,
    pub cell_a: usize,
    pub cell_b: usize,
    pub cell_c: usize,
    pub cell_d: usize,
}

fn edge_orthants(facet_m: usize, facet_n: usize, side_m: usize, side_n: usize) -> [usize; 2] {
    let axis_m = facet_m >> 1;
    let axis_n = facet_n >> 1;
    let axis_t = 3 - axis_m - axis_n;
    let base = (side_m << axis_m) | (side_n << axis_n);
    [base, base | (1 << axis_t)]
}

pub(crate) fn detect<T, U>(tree: &Octree<T, U>) -> Vec<EdgeMatch>
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    let child = |n: usize, o: usize| tree.nodes[n].orthants().unwrap()[o].into();
    let is_leaf = |n: usize, o: usize| tree.nodes[child(n, o)].is_leaf();
    let mut found = Vec::new();
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
                            && orthants_d
                                .into_iter()
                                .all(|p| tree.nodes[child(edge_child, p)].is_leaf())
                    })
                {
                    found.push(EdgeMatch {
                        facet_m,
                        facet_n,
                        cell_a,
                        cell_b,
                        cell_c,
                        cell_d,
                    });
                }
            }
        }
    }
    found
}

#[allow(clippy::too_many_arguments)]
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
    for edge_match in detect(tree) {
        let orth = |node: usize, o: usize| -> usize {
            tree.nodes[node].orthants().unwrap()[o].into()
        };
        let dir_m = facet_direction(edge_match.facet_m);
        let dir_n = facet_direction(edge_match.facet_n);
        let axis_t = 3 - (edge_match.facet_m >> 1) - (edge_match.facet_n >> 1);
        let dir_t = facet_direction(2 * axis_t + 1);
        let length: Scalar = tree.nodes[edge_match.cell_a].length.into();
        let half = 0.5 * length; // A half-length: A center -> edge line
        let fine = 0.125 * length; // D half-length: edge line -> finest centers
        // edge-line point at A's (facet_m, facet_n) corner
        let edge = &(&coordinates[center_nodes[edge_match.cell_a]] + &(&dir_m * half))
            + &(&dir_n * half);
        // D's two middle cells along the edge are existing dual nodes
        let od = edge_orthants(
            edge_match.facet_m,
            edge_match.facet_n,
            (edge_match.facet_m ^ 1) & 1,
            (edge_match.facet_n ^ 1) & 1,
        );
        let d_lo = center_nodes[orth(orth(edge_match.cell_d, od[0]), od[1])];
        let d_hi = center_nodes[orth(orth(edge_match.cell_d, od[1]), od[0])];
        // remaining six corners at the finest spacing around the edge line
        let corner = |sm: Scalar, sn: Scalar, st: Scalar| -> Coordinate<D> {
            &(&(&edge + &(&dir_m * (sm * fine))) + &(&dir_n * (sn * fine))) + &(&dir_t * (st * fine))
        };
        let a_lo = get_or_add(corner(-1.0, -1.0, -1.0), coordinates, nodes_map, node_index);
        let b_lo = get_or_add(corner(1.0, -1.0, -1.0), coordinates, nodes_map, node_index);
        let c_lo = get_or_add(corner(-1.0, 1.0, -1.0), coordinates, nodes_map, node_index);
        let a_hi = get_or_add(corner(-1.0, -1.0, 1.0), coordinates, nodes_map, node_index);
        let b_hi = get_or_add(corner(1.0, -1.0, 1.0), coordinates, nodes_map, node_index);
        let c_hi = get_or_add(corner(-1.0, 1.0, 1.0), coordinates, nodes_map, node_index);
        connectivity.push([a_lo, b_lo, d_lo, c_lo, a_hi, b_hi, d_hi, c_hi]);
    }
}
