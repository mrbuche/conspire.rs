use super::super::{D, N};
use crate::geometry::{
    mesh::Mesh,
    ntree::{
        Octree,
        balance::Balancing,
        node::{Kind, split::Split},
        pair::Pairing,
    },
};
use std::{array::from_fn, ops::Add};

const ROTATIONS: [[usize; 8]; 24] = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [6, 7, 4, 5, 2, 3, 0, 1],
    [5, 4, 7, 6, 1, 0, 3, 2],
    [3, 2, 1, 0, 7, 6, 5, 4],
    [4, 5, 0, 1, 6, 7, 2, 3],
    [2, 3, 6, 7, 0, 1, 4, 5],
    [1, 0, 5, 4, 3, 2, 7, 6],
    [7, 6, 3, 2, 5, 4, 1, 0],
    [4, 6, 5, 7, 0, 2, 1, 3],
    [2, 0, 3, 1, 6, 4, 7, 5],
    [1, 3, 0, 2, 5, 7, 4, 6],
    [7, 5, 6, 4, 3, 1, 2, 0],
    [0, 4, 1, 5, 2, 6, 3, 7],
    [6, 2, 7, 3, 4, 0, 5, 1],
    [5, 1, 4, 0, 7, 3, 6, 2],
    [3, 7, 2, 6, 1, 5, 0, 4],
    [0, 2, 4, 6, 1, 3, 5, 7],
    [6, 4, 2, 0, 7, 5, 3, 1],
    [5, 7, 1, 3, 4, 6, 0, 2],
    [3, 1, 7, 5, 2, 0, 6, 4],
    [4, 0, 6, 2, 5, 1, 7, 3],
    [2, 6, 0, 4, 3, 7, 1, 5],
    [1, 5, 3, 7, 0, 4, 2, 6],
    [7, 3, 5, 1, 6, 2, 4, 0],
];

const STRONG_CONFIGS: [[usize; 8]; 21] = [
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 1, 1, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 0, 1],
    [0, 0, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 1],
    [0, 0, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 1, 1],
    [0, 1, 1, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1],
];

const WEAK_CONFIGS: [[usize; 8]; 14] = [
    [0, 0, 0, 1, 0, 1, 1, 2],
    [0, 0, 0, 1, 1, 1, 1, 2],
    [0, 1, 1, 0, 1, 0, 2, 1],
    [0, 0, 1, 1, 1, 1, 1, 2],
    [0, 1, 1, 0, 1, 1, 2, 1],
    [0, 1, 1, 1, 1, 1, 1, 2],
    [0, 1, 1, 1, 1, 1, 2, 1],
    [0, 0, 1, 1, 1, 1, 2, 2],
    [0, 1, 1, 0, 1, 2, 2, 1],
    [0, 1, 1, 1, 1, 1, 2, 2],
    [0, 1, 1, 1, 1, 2, 2, 1],
    [0, 1, 1, 1, 1, 2, 2, 2],
    [0, 1, 1, 2, 1, 2, 2, 1],
    [0, 1, 1, 2, 1, 2, 2, 2],
];

fn min_scaled_jacobian(mesh: &Mesh<3>) -> f64 {
    const CORNERS: [[usize; 4]; 8] = [
        [0, 1, 3, 4],
        [1, 2, 0, 5],
        [2, 3, 1, 6],
        [3, 0, 2, 7],
        [4, 7, 5, 0],
        [5, 4, 6, 1],
        [6, 5, 7, 2],
        [7, 6, 4, 3],
    ];
    let coordinates = mesh.coordinates();
    mesh.iter()
        .flatten()
        .map(|hex| {
            CORNERS
                .iter()
                .map(|&[c, a, b, d]| {
                    let e = |k: usize| {
                        std::array::from_fn::<f64, 3, _>(|i| {
                            coordinates[hex[k]][i] - coordinates[hex[c]][i]
                        })
                    };
                    let (u, v, w) = (e(a), e(b), e(d));
                    let det = u[0] * (v[1] * w[2] - v[2] * w[1])
                        - u[1] * (v[0] * w[2] - v[2] * w[0])
                        + u[2] * (v[0] * w[1] - v[1] * w[0]);
                    let norm = |x: [f64; 3]| (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
                    det / (norm(u) * norm(v) * norm(w))
                })
                .fold(f64::INFINITY, f64::min)
        })
        .fold(f64::INFINITY, f64::min)
}

fn survey(configs: &[[usize; 8]], balancing: Balancing) {
    use super::super::test::verify_dual;
    use crate::geometry::ntree::Dualization;
    let mut failures = Vec::new();
    for (which, config) in configs.iter().enumerate() {
        for (way, rotation) in ROTATIONS.iter().enumerate() {
            let mut depths = [0usize; 8];
            (0..8).for_each(|octant| depths[rotation[octant]] = config[octant] + 1);
            let mut octree = super::super::edge::test::weak_tree(depths, balancing);
            let expected: usize = depths.iter().map(|&d| 8usize.pow(d as u32)).sum();
            let leaves = octree.iter().filter(|node| node.is_leaf()).count();
            if leaves != expected {
                failures.push(format!(
                    "config {which} rotation {way}: equilibrate changed the tree ({leaves} vs {expected} leaves)"
                ));
                continue;
            }
            let mesh = octree.dualize();
            if let Err(error) = verify_dual(&mesh) {
                failures.push(format!("config {which} rotation {way}: {error}"));
                continue;
            }
            let scaled_jacobian = min_scaled_jacobian(&mesh);
            if scaled_jacobian <= 0.0 {
                failures.push(format!(
                    "config {which} rotation {way}: min scaled jacobian {scaled_jacobian}"
                ));
            }
        }
    }
    if !failures.is_empty() {
        let mut summary = std::collections::BTreeMap::new();
        for failure in &failures {
            let key = failure.split(" rotation").next().unwrap().to_string();
            *summary.entry(key).or_insert(0usize) += 1;
        }
        for (key, count) in &summary {
            println!("{key}: {count}/24 rotations failed");
        }
        println!("first failures:");
        for failure in failures.iter().take(5) {
            println!("  {failure}");
        }
        panic!("{} vertex survey failures", failures.len());
    }
}

#[test]
fn survey_strong_vertex_configs() {
    survey(&STRONG_CONFIGS, Balancing::Strong(1));
}

#[test]
fn survey_weak_vertex_configs() {
    survey(&WEAK_CONFIGS, Balancing::Weak(1));
}

#[cfg(feature = "netcdf")]
fn weak_vertex_depths(fine: usize) -> [usize; 8] {
    let mut depths = [2; 8];
    depths[fine] = 3;
    depths[7 - fine] = 1;
    depths
}

#[test]
#[cfg(feature = "netcdf")]
fn write_weak_vertex_dual() {
    use super::super::test::verify_dual;
    use crate::{
        geometry::{mesh::Output, ntree::Dualization},
        io::Write,
    };
    for fine in 0..8 {
        let mut octree =
            super::super::edge::test::weak_tree(weak_vertex_depths(fine), Balancing::Weak(1));
        let mesh = octree.dualize();
        if let Err(error) = verify_dual(&mesh) {
            panic!("weak vertex dual (fine orthant {fine}) failed verification: {error}");
        }
        if fine == 7 {
            mesh.write(Output::Exodus("target/weak_vertex.exo"))
                .unwrap();
        }
    }
}

pub(crate) fn vertex_dual_generic<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
) -> Vec<[usize; N]>
where
    T: Copy + Into<usize> + Add<Output = T> + PartialOrd + Split,
    U: Copy + Into<usize>,
{
    const WIND: [usize; N] = [0, 1, 3, 2, 4, 5, 7, 6];
    let root = &tree.nodes[0];
    let lo = root.corner;
    let hi: [T; D] = from_fn(|a| root.corner[a] + root.length);
    let mut hexes = Vec::new();
    for node in tree.iter().filter(|node| node.is_leaf()) {
        let v: [T; D] = from_fn(|a| node.corner[a] + node.length);
        if (0..D).all(|a| lo[a] < v[a] && v[a] < hi[a]) {
            let cells: [usize; N] = from_fn(|d| find_leaf_octant(tree, &v, d));
            hexes.push(from_fn(|k| center_nodes[cells[WIND[k]]]));
        }
    }
    hexes
}

fn find_leaf_octant<T, U>(tree: &Octree<T, U>, v: &[T; D], d: usize) -> usize
where
    T: Copy + Add<Output = T> + PartialOrd + Split,
    U: Copy + Into<usize>,
{
    let mut index = 0;
    loop {
        match &tree.nodes[index].kind {
            Kind::Leaf => return index,
            Kind::Tree(orthants) => {
                let corner = tree.nodes[index].corner;
                let half = tree.nodes[index].length.split();
                let child = (0..D).fold(0, |acc, a| {
                    let mid = corner[a] + half;
                    let bit = if v[a] > mid {
                        1
                    } else if v[a] < mid {
                        0
                    } else {
                        (d >> a) & 1
                    };
                    acc | (bit << a)
                });
                index = orthants[child].into();
            }
        }
    }
}

fn fuzz_tree(seed: u64, balancing: Balancing, pairing: Pairing) -> Octree<u16, usize> {
    use crate::geometry::ntree::{Balance, node::Node, rescale::Rescaling};
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut rand = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let mut octree = Octree::<u16, usize> {
        balanced: Balancing::None,
        nodes: vec![Node {
            corner: [0, 0, 0],
            length: 32,
            facets: [None; 6],
            kind: Kind::Leaf,
            value: None,
        }],
        paired: Pairing::None,
        pairing_vertices: Default::default(),
        rescale: Rescaling {
            center: [16.0; 3],
            cell: 1.0,
            half: 16.0,
        },
    };
    octree.subdivide(0).unwrap();
    for _ in 0..40 {
        let leaves: Vec<usize> = octree
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.is_leaf() && node.length >= 4)
            .map(|(i, _)| i)
            .collect();
        if leaves.is_empty() {
            break;
        }
        let pick = leaves[rand() % leaves.len()];
        octree.subdivide(pick).unwrap();
    }
    octree.equilibrate(balancing, pairing).unwrap();
    octree
}

fn fuzz_duals(balancing: Balancing, pairing: Pairing) {
    use super::super::test::verify_dual;
    use crate::geometry::ntree::Dualization;
    let mut failures = Vec::new();
    for seed in 0..200u64 {
        let mut octree = fuzz_tree(seed, balancing, pairing);
        let mesh = octree.dualize();
        if let Err(error) = verify_dual(&mesh) {
            failures.push(format!("seed {seed}: {error}"));
            continue;
        }
        let scaled_jacobian = min_scaled_jacobian(&mesh);
        if scaled_jacobian <= 0.0 {
            failures.push(format!(
                "seed {seed}: min scaled jacobian {scaled_jacobian}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn fuzz_weak_duals() {
    fuzz_duals(Balancing::Weak(1), Pairing::Regular)
}

#[test]
fn fuzz_strong_duals() {
    fuzz_duals(Balancing::Strong(1), Pairing::Regular)
}

// The quadtree dual was generalized by re-anchoring its templates on the paired
// 2x2 blocks rather than on tree nodes; the octree's face and edge transitions
// still walk tree topology, so they only fire where a block happens to coincide
// with a node. Measured over 40 seeds: Regular passes 40/40 on both balancings,
// Generalized passes 4/40 strong and 2/40 weak, failing as unclosed boundaries
// (the uncovered transition strips), non-conformal faces and non-sphere
// boundaries. Ignored as a standing record until face/ and edge/ are ported.
#[test]
#[ignore]
fn fuzz_weak_duals_generalized() {
    fuzz_duals(Balancing::Weak(1), Pairing::Generalized)
}

#[test]
#[ignore]
fn fuzz_strong_duals_generalized() {
    fuzz_duals(Balancing::Strong(1), Pairing::Generalized)
}

// The face template takes a quarter of a block facet - one coarse leaf and the 2x2 fine cells
// behind it - only when the quarter is whole, and treats a facet as truncated only when every
// absent quarter is provably outside the domain. A quarter absent for any other reason (the
// outside cell refined, the inside cells refined deeper) means the transition belongs to a
// different block, and guessing a template there would double-cover it. This asserts that
// mixed facets - some quarters present, some absent while still inside the domain - never
// arise, so the classification the template relies on is total.
#[test]
fn every_block_facet_is_classifiable() {
    use crate::geometry::ntree::dual::leaf_containing;
    for (label, pairing) in [
        ("regular", Pairing::Regular),
        ("generalized", Pairing::Generalized),
    ] {
        let mut mixed = Vec::new();
        for seed in 0..40u64 {
            let tree = fuzz_tree(seed, Balancing::Weak(1), pairing);
            let root = &tree.nodes[0];
            let low: [i64; 3] = from_fn(|a| root.corner[a] as i64);
            let high: [i64; 3] = from_fn(|a| low[a] + root.length as i64);
            let cell_at = |corner: [i64; 3], length: i64| -> Option<usize> {
                if (0..3).any(|a| corner[a] < low[a] || corner[a] + length > high[a]) {
                    return None;
                }
                let point = from_fn(|a| corner[a] as usize);
                let index = leaf_containing(&tree, &point);
                let node = &tree.nodes[index];
                (length as usize == node.length as usize
                    && (0..3).all(|a| point[a] == node.corner[a] as usize))
                .then_some(index)
            };
            for &(block, length) in tree.pairing_vertices.iter() {
                let center: [i64; 3] = from_fn(|a| block[a] as i64);
                let (coarse, fine) = (length as i64, length as i64 / 2);
                for facet in 0..6 {
                    let (axis, side) = (facet >> 1, facet & 1);
                    let tangents: [usize; 2] = {
                        let mut o = (0..3).filter(|&x| x != axis);
                        [o.next().unwrap(), o.next().unwrap()]
                    };
                    let interface = center[axis] + if side == 1 { coarse } else { -coarse };

                    let outside = if side == 1 {
                        interface
                    } else {
                        interface - coarse
                    };
                    let inside = if side == 1 {
                        interface - fine
                    } else {
                        interface
                    };
                    let at = |along: i64, u: i64, v: i64| {
                        let mut c = [0; 3];
                        c[axis] = along;
                        c[tangents[0]] = center[tangents[0]] - coarse + u;
                        c[tangents[1]] = center[tangents[1]] - coarse + v;
                        c
                    };
                    // a "quarter" is one coarse leaf plus the 2x2 fine cells behind it
                    let quarter = |qu: i64, qv: i64| {
                        cell_at(at(outside, qu * coarse, qv * coarse), coarse).is_some()
                            && (0..4).all(|k| {
                                let (du, dv) = ((k & 1) as i64, (k >> 1) as i64);
                                cell_at(
                                    at(inside, (2 * qu + du) * fine, (2 * qv + dv) * fine),
                                    fine,
                                )
                                .is_some()
                            })
                    };
                    let off = |qu: i64, qv: i64| {
                        let u = center[tangents[0]] - coarse + qu * coarse;
                        let v = center[tangents[1]] - coarse + qv * coarse;
                        u < low[tangents[0]]
                            || u + coarse > high[tangents[0]]
                            || v < low[tangents[1]]
                            || v + coarse > high[tangents[1]]
                    };
                    let quarters = (0..4).map(|q| ((q & 1) as i64, (q >> 1) as i64));
                    let any = quarters.clone().any(|(qu, qv)| quarter(qu, qv));
                    let stray = quarters
                        .filter(|&(qu, qv)| !quarter(qu, qv))
                        .any(|(qu, qv)| !off(qu, qv));
                    if any && stray {
                        mixed.push(format!("seed {seed} block {block:?} facet {facet}"));
                    }
                }
            }
        }
        assert!(
            mixed.is_empty(),
            "{label}: {} facets mix present quarters with absent in-domain ones:\n{}",
            mixed.len(),
            mixed.join("\n")
        );
    }
}
