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
        let EdgeMatch {
            facet_m,
            facet_n,
            cell_a,
            cell_b,
            cell_c,
            cell_d,
        } = edge_match;
        let orth = |node: usize, o: usize| -> usize { tree.nodes[node].orthants().unwrap()[o].into() };
        let dir_m = facet_direction(facet_m);
        let dir_n = facet_direction(facet_n);
        let axis_t = 3 - (facet_m >> 1) - (facet_n >> 1);
        let dir_t = facet_direction(2 * axis_t + 1);
        let length: Scalar = tree.nodes[cell_a].length.into();
        let half = 0.5 * length; // A center -> edge line
        let quad = 0.25 * length; // B/C (level-1) half-length
        let fine = 0.125 * length; // D (level-2) half-length
        let edge = &(&coordinates[center_nodes[cell_a]] + &(&dir_m * half)) + &(&dir_n * half);
        let point = |sm: Scalar, sn: Scalar, st: Scalar, scale: Scalar| -> Coordinate<D> {
            &(&(&edge + &(&dir_m * (sm * scale))) + &(&dir_n * (sn * scale)))
                + &(&dir_t * (st * scale))
        };
        // existing dual centers: A, the two middle D cells, and B/C edge cells
        let od = edge_orthants(facet_m, facet_n, (facet_m ^ 1) & 1, (facet_n ^ 1) & 1);
        let ob = edge_orthants(facet_m, facet_n, (facet_m ^ 1) & 1, facet_n & 1);
        let oc = edge_orthants(facet_m, facet_n, facet_m & 1, (facet_n ^ 1) & 1);
        let node_a = center_nodes[cell_a];
        let d_lo = center_nodes[orth(orth(cell_d, od[0]), od[1])];
        let d_hi = center_nodes[orth(orth(cell_d, od[1]), od[0])];
        let b_lo_c = center_nodes[orth(cell_b, ob[0])];
        let b_hi_c = center_nodes[orth(cell_b, ob[1])];
        let c_lo_c = center_nodes[orth(cell_c, oc[0])];
        let c_hi_c = center_nodes[orth(cell_c, oc[1])];
        // finest-spacing ring around the edge line (shared with the central cube)
        let a_lo = get_or_add(point(-1.0, -1.0, -1.0, fine), coordinates, nodes_map, node_index);
        let b_lo = get_or_add(point(1.0, -1.0, -1.0, fine), coordinates, nodes_map, node_index);
        let c_lo = get_or_add(point(-1.0, 1.0, -1.0, fine), coordinates, nodes_map, node_index);
        let a_hi = get_or_add(point(-1.0, -1.0, 1.0, fine), coordinates, nodes_map, node_index);
        let b_hi = get_or_add(point(1.0, -1.0, 1.0, fine), coordinates, nodes_map, node_index);
        let c_hi = get_or_add(point(-1.0, 1.0, 1.0, fine), coordinates, nodes_map, node_index);
        // level-1 lattice point diagonally toward A, shared by both lateral hexes
        let steiner = get_or_add(point(-1.0, -1.0, -1.0, quad), coordinates, nodes_map, node_index);
        // central cube
        connectivity.push([a_lo, b_lo, d_lo, c_lo, a_hi, b_hi, d_hi, c_hi]);
        // lateral hex toward B (-n side)
        connectivity.push([a_lo, steiner, b_lo_c, b_lo, a_hi, node_a, b_hi_c, b_hi]);
        // lateral hex toward C (-m side)
        connectivity.push([a_lo, c_lo, c_lo_c, steiner, a_hi, c_hi, c_hi_c, node_a]);
    }
}
