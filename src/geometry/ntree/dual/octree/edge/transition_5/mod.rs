use crate::{
    geometry::{
        Coordinate, Coordinates,
        ntree::{
            Octree,
            dual::{
                NodeMap, get_or_add,
                octree::{D, M, N, facet_direction},
            },
            node::Node,
        },
    },
    math::Scalar,
};

// the 12 cube edges as facet pairs on different axes
const EDGES: [(usize, usize); 12] = [
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 4),
    (2, 5),
    (3, 4),
    (3, 5),
];

// the two orthants straddling the edge along the third axis, given the sides
// each facet presents toward the edge
fn edge_orthants(facet_m: usize, facet_n: usize, side_m: usize, side_n: usize) -> [usize; 2] {
    let axis_m = facet_m >> 1;
    let axis_n = facet_n >> 1;
    let axis_t = 3 - axis_m - axis_n;
    let base = (side_m << axis_m) | (side_n << axis_n);
    [base, base | (1 << axis_t)]
}

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
    for (cell_a, node) in tree.iter().enumerate() {
        for &(facet_m, facet_n) in EDGES.iter() {
            template_inner(
                cell_a,
                facet_m,
                facet_n,
                node,
                tree,
                center_nodes,
                coordinates,
                connectivity,
                node_index,
                nodes_map,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn template_inner<T, U>(
    cell_a: usize,
    mut facet_m: usize,
    mut facet_n: usize,
    node: &Node<D, M, N, T, U>,
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
    let orth = |cell: usize, o: usize| -> usize { tree.nodes[cell].orthants().unwrap()[o].into() };
    if !node.is_leaf() {
        return;
    }
    let (Some(neighbor_m), Some(neighbor_n)) = (node.facets[facet_m], node.facets[facet_n]) else {
        return;
    };
    let (mut cell_b, mut cell_c) = (neighbor_m.into(), neighbor_n.into());
    if !tree.nodes[cell_b].is_tree() || !tree.nodes[cell_c].is_tree() {
        return;
    }
    let Some(neighbor_diag) = tree.nodes[cell_b].facets[facet_n] else {
        return;
    };
    let cell_d: usize = neighbor_diag.into();
    if !tree.nodes[cell_d].is_tree() {
        return;
    }
    // level pinning: B,C edge children leaves (level 1); D edge children trees
    // with leaf grandchildren (level 2)
    let edge_b = edge_orthants(facet_m, facet_n, (facet_m ^ 1) & 1, facet_n & 1);
    let edge_c = edge_orthants(facet_m, facet_n, facet_m & 1, (facet_n ^ 1) & 1);
    let edge_d = edge_orthants(facet_m, facet_n, (facet_m ^ 1) & 1, (facet_n ^ 1) & 1);
    if !edge_b.into_iter().all(|o| tree.nodes[orth(cell_b, o)].is_leaf())
        || !edge_c.into_iter().all(|o| tree.nodes[orth(cell_c, o)].is_leaf())
        || !edge_d.into_iter().all(|o| {
            let child = orth(cell_d, o);
            tree.nodes[child].is_tree() && edge_d.into_iter().all(|p| tree.nodes[orth(child, p)].is_leaf())
        })
    {
        return;
    }
    // keep (dir_m, dir_n, +t) right-handed so one winding serves every orientation
    let axis_t = 3 - (facet_m >> 1) - (facet_n >> 1);
    let dir_t = facet_direction(2 * axis_t + 1);
    let (m, n) = (facet_direction(facet_m), facet_direction(facet_n));
    let triple = m[0] * (n[1] * dir_t[2] - n[2] * dir_t[1])
        - m[1] * (n[0] * dir_t[2] - n[2] * dir_t[0])
        + m[2] * (n[0] * dir_t[1] - n[1] * dir_t[0]);
    if triple < 0.0 {
        std::mem::swap(&mut facet_m, &mut facet_n);
        std::mem::swap(&mut cell_b, &mut cell_c);
    }
    let edge_b = edge_orthants(facet_m, facet_n, (facet_m ^ 1) & 1, facet_n & 1);
    let edge_c = edge_orthants(facet_m, facet_n, facet_m & 1, (facet_n ^ 1) & 1);
    let edge_d = edge_orthants(facet_m, facet_n, (facet_m ^ 1) & 1, (facet_n ^ 1) & 1);
    let dir_m = facet_direction(facet_m);
    let dir_n = facet_direction(facet_n);
    let length: Scalar = tree.nodes[cell_a].length.into();
    let half = &(&dir_m * (0.5 * length)) + &(&dir_n * (0.5 * length)); // A center -> edge line
    let edge = &coordinates[center_nodes[cell_a]] + &half;
    let off_m = &dir_m * (0.125 * length); // finest (D) half-step
    let off_n = &dir_n * (0.125 * length);
    let off_t = &dir_t * (0.125 * length);
    let diag = &(&(&dir_m * (0.25 * length)) + &(&dir_n * (0.25 * length))) + &(&dir_t * (0.25 * length));
    // existing dual centers
    let node_a = center_nodes[cell_a];
    let d_lo = center_nodes[orth(orth(cell_d, edge_d[0]), edge_d[1])];
    let d_hi = center_nodes[orth(orth(cell_d, edge_d[1]), edge_d[0])];
    let b_lo = center_nodes[orth(cell_b, edge_b[0])];
    let b_hi = center_nodes[orth(cell_b, edge_b[1])];
    let c_lo = center_nodes[orth(cell_c, edge_c[0])];
    let c_hi = center_nodes[orth(cell_c, edge_c[1])];
    // finest-spacing ring around the edge line, plus the level-1 point toward A
    let [ring_a_lo, ring_b_lo, ring_c_lo, ring_a_hi, ring_b_hi, ring_c_hi] = [
        &(&(&edge - &off_m) - &off_n) - &off_t,
        &(&(&edge + &off_m) - &off_n) - &off_t,
        &(&(&edge - &off_m) + &off_n) - &off_t,
        &(&(&edge - &off_m) - &off_n) + &off_t,
        &(&(&edge + &off_m) - &off_n) + &off_t,
        &(&(&edge - &off_m) + &off_n) + &off_t,
    ]
    .map(|coordinate| get_or_add(coordinate, coordinates, nodes_map, node_index));
    let steiner = get_or_add(&edge - &diag, coordinates, nodes_map, node_index);
    // central cube
    connectivity.push([ring_a_lo, ring_b_lo, d_lo, ring_c_lo, ring_a_hi, ring_b_hi, d_hi, ring_c_hi]);
    // lateral hex toward B
    connectivity.push([ring_a_lo, steiner, b_lo, ring_b_lo, ring_a_hi, node_a, b_hi, ring_b_hi]);
    // lateral hex toward C
    connectivity.push([ring_a_lo, ring_c_lo, c_lo, steiner, ring_a_hi, ring_c_hi, c_hi, node_a]);
}
