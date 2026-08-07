use crate::{
    geometry::{
        Coordinate, Coordinates,
        ntree::{
            Octree,
            dual::{
                NodeMap, get_or_add,
                octree::{D, N, facet_direction},
            },
            node::{Node, split::Split},
        },
    },
    math::Scalar,
};
use std::array::from_fn;

const M: usize = 6;

const EDGES: [(usize, usize); 12] = [
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

struct Config {
    center: usize,
    length: Scalar,
    n_lo: usize,
    n_hi: usize,
    m_lo: usize,
    m_hi: usize,
    ring_lo: usize,
    ladder_lo: usize,
    ladder_hi: usize,
    ring_hi: usize,
}

pub(super) fn template<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    coordinates: &mut Coordinates<D>,
    connectivity: &mut Vec<[usize; N]>,
    node_index: &mut usize,
    nodes_map: &mut NodeMap<D>,
) where
    T: Copy + Into<Scalar> + Into<usize> + Split,
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
                let axis = 3 - (facet_m >> 1) - (facet_n >> 1);
                // The tree-walking version asked whether this leaf was the low half of an
                // aligned pair, which is to say whether it and the leaf above are siblings.
                // Siblinghood is really "the pair is paired", so read it off the refined
                // columns beside them, the way the neighbouring wedge does - a coarse leaf
                // belongs to no cluster itself, but its refined neighbours do.
                let origin: [i64; D] = from_fn(|a| Into::<usize>::into(node.corner[a]) as i64);
                let coarse = Into::<usize>::into(node.length) as i64;
                let column = |facets: &[usize]| {
                    let mut corner = origin;
                    for &facet in facets {
                        corner[facet >> 1] += if facet & 1 == 1 { coarse } else { -coarse };
                    }
                    corner
                };
                let mut upper = origin;
                upper[axis] += coarse;
                if tree.shares_cluster(&column(&[facet_m]), coarse, axis)
                    && tree.shares_cluster(&column(&[facet_n]), coarse, axis)
                    && tree.shares_cluster(&column(&[facet_m, facet_n]), coarse, axis)
                    && let Some(above) = tree.cell_at(&upper, coarse)
                    && let Some(config_b) = config(
                        tree,
                        &tree.nodes[above],
                        above,
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

fn config<T, U>(
    tree: &Octree<T, U>,
    node: &Node<D, M, N, T, U>,
    index: usize,
    facet_m: usize,
    facet_n: usize,
    center_nodes: &[usize],
) -> Option<Config>
where
    T: Copy + Into<Scalar> + Into<usize> + Split,
    U: Copy + Into<usize>,
{
    let axis_m = facet_m >> 1;
    let axis_n = facet_n >> 1;
    let axis = 3 - axis_m - axis_n;
    let side_m = facet_m & 1;
    let side_n = facet_n & 1;
    // The two adjoining neighbours are one level finer than this leaf and the diagonal one is
    // two levels finer, which is the 4:1 jump only `Weak` balancing permits. Every cell below is
    // therefore fixed by position alone; walking `facets` and `leaves` to reach them only worked
    // while clusters coincided with nodes.
    let coarse = Into::<usize>::into(node.length) as i64;
    let (half, quarter) = (coarse / 2, coarse / 4);
    if quarter == 0 {
        return None;
    }
    let origin: [i64; D] = from_fn(|a| Into::<usize>::into(node.corner[a]) as i64);
    let corner_at = |offset_m: i64, offset_n: i64, offset: i64| {
        let mut corner = origin;
        corner[axis_m] += offset_m;
        corner[axis_n] += offset_n;
        corner[axis] += offset;
        corner
    };
    let (beyond_m, beside_m) = (
        if side_m == 1 { coarse } else { -half },
        side_m as i64 * half,
    );
    let (beyond_n, beside_n) = (
        if side_n == 1 { coarse } else { -half },
        side_n as i64 * half,
    );
    let (far_m, far_n) = (
        if side_m == 1 { coarse } else { -quarter },
        if side_n == 1 { coarse } else { -quarter },
    );
    let m_lo = tree.cell_at(&corner_at(beyond_m, beside_n, 0), half)?;
    let m_hi = tree.cell_at(&corner_at(beyond_m, beside_n, half), half)?;
    let n_lo = tree.cell_at(&corner_at(beside_m, beyond_n, 0), half)?;
    let n_hi = tree.cell_at(&corner_at(beside_m, beyond_n, half), half)?;
    // Four cells along the edge on the diagonal; the middle two carry the Steiner ladder.
    let rungs: [usize; 4] = from_fn(|k| {
        tree.cell_at(&corner_at(far_m, far_n, k as i64 * quarter), quarter)
            .unwrap_or(usize::MAX)
    });
    if rungs.contains(&usize::MAX) {
        return None;
    }
    let [ring_lo, ladder_lo, ladder_hi, ring_hi] = rungs;
    Some(Config {
        center: center_nodes[index],
        length: tree.nodes[ring_lo].length.into(),
        n_lo: center_nodes[n_lo],
        n_hi: center_nodes[n_hi],
        m_lo: center_nodes[m_lo],
        m_hi: center_nodes[m_hi],
        ring_lo: center_nodes[ring_lo],
        ladder_lo: center_nodes[ladder_lo],
        ladder_hi: center_nodes[ladder_hi],
        ring_hi: center_nodes[ring_hi],
    })
}

fn find(coordinate: &Coordinate<D>, nodes_map: &NodeMap<D>) -> Option<usize> {
    nodes_map
        .get(&from_fn(|i| (2.0 * coordinate[i]) as usize))
        .copied()
}

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
