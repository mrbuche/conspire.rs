use crate::{
    geometry::{
        Coordinate, Coordinates,
        ntree::{
            Octree,
            dual::{
                NodeMap, get_or_add,
                octree::{D, N, facet_direction},
            },
            node::Node,
        },
    },
    math::Scalar,
};
use std::array::from_fn;

const M: usize = 6;

// (facet_m, facet_n) per edge orientation, ordered so that the frame
// (facet_m direction, facet_n direction, +edge axis) is right-handed.
pub(crate) const EDGES: [(usize, usize); 12] = [
    (1, 3),
    (2, 1),
    (3, 0),
    (0, 2),
    (3, 5),
    (4, 3),
    (5, 2),
    (2, 4),
    (5, 1),
    (0, 5),
    (1, 4),
    (4, 0),
];

// Weak (face-balanced only) edge configuration around one coarse cell: the
// facet_m and facet_n neighbors hold half-size leaves, whose facet_n
// neighbors across the edge are trees holding quarter-size leaves.
pub(crate) struct Config {
    pub(crate) center: usize,
    pub(crate) length: Scalar,
    pub(crate) n_lo: usize,
    pub(crate) n_hi: usize,
    pub(crate) m_lo: usize,
    pub(crate) m_hi: usize,
    pub(crate) ring_lo: usize,
    pub(crate) ladder_lo: usize,
    pub(crate) ladder_hi: usize,
    pub(crate) ring_hi: usize,
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
    for (index, node) in tree.nodes.iter().enumerate() {
        if !node.is_leaf() {
            continue;
        }
        for &(facet_m, facet_n) in EDGES.iter() {
            if let Some(config_a) = config(tree, node, index, facet_m, facet_n, center_nodes) {
                chamber(
                    &config_a,
                    facet_m,
                    facet_n,
                    coordinates,
                    connectivity,
                    node_index,
                    nodes_map,
                );
                // The pair hex across a within-parent face; a pair meeting at
                // a parent grid plane is a vertex configuration instead
                // (handled by vertex transition_22).
                let axis = 3 - (facet_m >> 1) - (facet_n >> 1);
                let corner: usize = node.corner[axis].into();
                let length: usize = node.length.into();
                if !(corner + length).is_multiple_of(2 * length)
                    && let Some(above) = node.facets[2 * axis + 1]
                    && tree.nodes[above.into()].is_leaf()
                    && let Some(config_b) = config(
                        tree,
                        &tree.nodes[above.into()],
                        above.into(),
                        facet_m,
                        facet_n,
                        center_nodes,
                    )
                {
                    pair(
                        &config_a,
                        &config_b,
                        facet_m,
                        facet_n,
                        coordinates,
                        connectivity,
                        nodes_map,
                    );
                }
            }
        }
    }
}

pub(crate) fn config<T, U>(
    tree: &Octree<T, U>,
    node: &Node<D, M, N, T, U>,
    index: usize,
    facet_m: usize,
    facet_n: usize,
    center_nodes: &[usize],
) -> Option<Config>
where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    let axis_m = facet_m >> 1;
    let axis_n = facet_n >> 1;
    let axis = 3 - axis_m - axis_n;
    let side_m = facet_m & 1;
    let side_n = facet_n & 1;
    let c = ((1 - side_m) << axis_m) | (side_n << axis_n);
    let e = c | (1 << axis);
    let d = (side_m << axis_m) | ((1 - side_n) << axis_n);
    let f = d | (1 << axis);
    let g = ((1 - side_m) << axis_m) | ((1 - side_n) << axis_n);
    let g_hi = g | (1 << axis);
    let tree_m = node.facets[facet_m]?;
    let tree_n = node.facets[facet_n]?;
    let leaves_m = tree.leaves(&tree.nodes[tree_m.into()]);
    let leaves_n = tree.leaves(&tree.nodes[tree_n.into()]);
    let m_lo = leaves_m[c]?;
    let m_hi = leaves_m[e]?;
    let n_lo = leaves_n[d]?;
    let n_hi = leaves_n[f]?;
    let diagonal_lo = tree.nodes[m_lo.into()].facets[facet_n]?;
    let diagonal_hi = tree.nodes[m_hi.into()].facets[facet_n]?;
    let leaves_lo = tree.leaves(&tree.nodes[diagonal_lo.into()]);
    let leaves_hi = tree.leaves(&tree.nodes[diagonal_hi.into()]);
    let ring_lo = leaves_lo[g]?;
    let ladder_lo = leaves_lo[g_hi]?;
    let ladder_hi = leaves_hi[g]?;
    let ring_hi = leaves_hi[g_hi]?;
    Some(Config {
        center: center_nodes[index],
        length: tree.nodes[ring_lo.into()].length.into(),
        n_lo: center_nodes[n_lo.into()],
        n_hi: center_nodes[n_hi.into()],
        m_lo: center_nodes[m_lo.into()],
        m_hi: center_nodes[m_hi.into()],
        ring_lo: center_nodes[ring_lo.into()],
        ladder_lo: center_nodes[ladder_lo.into()],
        ladder_hi: center_nodes[ladder_hi.into()],
        ring_hi: center_nodes[ring_hi.into()],
    })
}

fn find(coordinate: &Coordinate<D>, nodes_map: &NodeMap<D>) -> Option<usize> {
    nodes_map
        .get(&from_fn(|i| (2.0 * coordinate[i]) as usize))
        .copied()
}

// Ring corner at one end of a chamber: the face-transition Steiner node if
// one exists there (within-parent pair face), otherwise the cell center
// itself (cross-parent pair face or domain boundary), along with the
// interior Steiner node for that end.
#[allow(clippy::too_many_arguments)]
fn corner(
    steiner: Coordinate<D>,
    config: &Config,
    center: &Coordinate<D>,
    diagonal: &Coordinate<D>,
    inward: Coordinate<D>,
    coordinates: &mut Coordinates<D>,
    nodes_map: &mut NodeMap<D>,
    node_index: &mut usize,
) -> (usize, usize) {
    if let Some(corner) = find(&steiner, nodes_map) {
        let interior = &steiner + &(&(diagonal + &inward) * 0.5);
        (
            corner,
            get_or_add(interior, coordinates, nodes_map, node_index),
        )
    } else {
        (
            config.center,
            get_or_add(center + diagonal, coordinates, nodes_map, node_index),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn chamber(
    config: &Config,
    facet_m: usize,
    facet_n: usize,
    coordinates: &mut Coordinates<D>,
    connectivity: &mut Vec<[usize; N]>,
    node_index: &mut usize,
    nodes_map: &mut NodeMap<D>,
) {
    let axis = 3 - (facet_m >> 1) - (facet_n >> 1);
    let length = config.length;
    let offset_m = &facet_direction(facet_m) * length;
    let offset_n = &facet_direction(facet_n) * length;
    let offset_up = &facet_direction(2 * axis + 1) * length;
    let ladder_lo = coordinates[config.ladder_lo].clone();
    let ladder_hi = coordinates[config.ladder_hi].clone();
    let (Some(pn_lo), Some(pm_lo), Some(pn_hi), Some(pm_hi)) = (
        find(&(&ladder_lo - &offset_m), nodes_map),
        find(&(&ladder_lo - &offset_n), nodes_map),
        find(&(&ladder_hi - &offset_m), nodes_map),
        find(&(&ladder_hi - &offset_n), nodes_map),
    ) else {
        return;
    };
    let center = coordinates[config.center].clone();
    let diagonal = &offset_m + &offset_n;
    let (x_lo, t_lo) = corner(
        &(&center + &diagonal) - &offset_up,
        config,
        &center,
        &diagonal,
        offset_up.clone(),
        coordinates,
        nodes_map,
        node_index,
    );
    let (x_hi, t_hi) = corner(
        &(&center + &diagonal) + &offset_up,
        config,
        &center,
        &diagonal,
        -offset_up,
        coordinates,
        nodes_map,
        node_index,
    );
    if x_lo == x_hi {
        return;
    }
    connectivity.push([
        t_hi,
        pn_hi,
        config.ladder_hi,
        pm_hi,
        t_lo,
        pn_lo,
        config.ladder_lo,
        pm_lo,
    ]);
    connectivity.push([
        t_lo,
        pn_lo,
        config.ladder_lo,
        pm_lo,
        x_lo,
        config.n_lo,
        config.ring_lo,
        config.m_lo,
    ]);
    connectivity.push([
        x_hi,
        config.n_hi,
        config.ring_hi,
        config.m_hi,
        t_hi,
        pn_hi,
        config.ladder_hi,
        pm_hi,
    ]);
    connectivity.push([
        x_hi,
        config.n_hi,
        pn_hi,
        t_hi,
        x_lo,
        config.n_lo,
        pn_lo,
        t_lo,
    ]);
    connectivity.push([
        x_hi,
        t_hi,
        pm_hi,
        config.m_hi,
        x_lo,
        t_lo,
        pm_lo,
        config.m_lo,
    ]);
}

#[allow(clippy::too_many_arguments)]
fn pair(
    config_a: &Config,
    config_b: &Config,
    facet_m: usize,
    facet_n: usize,
    coordinates: &Coordinates<D>,
    connectivity: &mut Vec<[usize; N]>,
    nodes_map: &NodeMap<D>,
) {
    if config_a.length != config_b.length {
        return;
    }
    let axis = 3 - (facet_m >> 1) - (facet_n >> 1);
    let length = config_a.length;
    let diagonal = &(&facet_direction(facet_m) + &facet_direction(facet_n)) * length;
    let offset_up = &facet_direction(2 * axis + 1) * length;
    let x_a = find(
        &(&(&coordinates[config_a.center] + &diagonal) + &offset_up),
        nodes_map,
    )
    .unwrap_or(config_a.center);
    let x_b = find(
        &(&(&coordinates[config_b.center] + &diagonal) - &offset_up),
        nodes_map,
    )
    .unwrap_or(config_b.center);
    connectivity.push([
        x_b,
        config_b.n_lo,
        config_b.ring_lo,
        config_b.m_lo,
        x_a,
        config_a.n_hi,
        config_a.ring_hi,
        config_a.m_hi,
    ]);
}
