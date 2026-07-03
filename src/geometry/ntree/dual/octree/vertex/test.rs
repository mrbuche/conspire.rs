use super::super::{D, N};
use super::{Template, apply, transition_21};
use crate::geometry::ntree::{
    Octree,
    balance::Balancing,
    node::{Kind, split::Split},
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

#[test]
fn knockout_strong_vertex_templates() {
    use super::super::test::verify_dual;
    use crate::geometry::{
        mesh::{Connectivity, Mesh},
        ntree::dual::{NodeMap, Uniform},
    };
    let templates: [super::Entry<u16, usize>; 21] = [
        (&super::transition_1::DATA, super::transition_1::template),
        (&super::transition_2::DATA, super::transition_2::template),
        (&super::transition_3::DATA, super::transition_3::template),
        (&super::transition_4::DATA, super::transition_4::template),
        (&super::transition_5::DATA, super::transition_5::template),
        (&super::transition_6::DATA, super::transition_6::template),
        (&super::transition_7::DATA, super::transition_7::template),
        (&super::transition_8::DATA, super::transition_8::template),
        (&super::transition_9::DATA, super::transition_9::template),
        (&super::transition_10::DATA, super::transition_10::template),
        (&super::transition_11::DATA, super::transition_11::template),
        (&super::transition_12::DATA, super::transition_12::template),
        (&super::transition_13::DATA, super::transition_13::template),
        (&super::transition_14::DATA, super::transition_14::template),
        (&super::transition_15::DATA, super::transition_15::template),
        (&super::transition_16::DATA, super::transition_16::template),
        (&super::transition_17::DATA, super::transition_17::template),
        (&super::transition_18::DATA, super::transition_18::template),
        (&super::transition_19::DATA, super::transition_19::template),
        (&super::transition_20::DATA, super::transition_20::template),
        (&super::transition_21::DATA, super::transition_21::template),
    ];
    let mut fired = [0usize; 21];
    let mut broke = [0usize; 21];
    for config in STRONG_CONFIGS.iter() {
        for rotation in ROTATIONS.iter() {
            let mut depths = [0usize; 8];
            (0..8).for_each(|octant| depths[rotation[octant]] = config[octant] + 1);
            let octree = super::super::edge::test::weak_tree(depths, Balancing::Strong);
            let (center_nodes, mut coordinates, mut node_index, mut connectivity) =
                octree.initialize();
            octree.uniform_transitions(&center_nodes, &mut connectivity);
            let mut nodes_map = NodeMap::new();
            super::super::face::face_transition(
                &octree,
                &center_nodes,
                &mut coordinates,
                &mut connectivity,
                &mut node_index,
                &mut nodes_map,
            );
            super::super::edge::edge_transitions(
                &octree,
                &center_nodes,
                &mut coordinates,
                &mut connectivity,
                &mut node_index,
                &mut nodes_map,
                Balancing::Strong,
            );
            let counts: Vec<usize> = templates
                .iter()
                .map(|&(data, template)| {
                    let before = connectivity.len();
                    apply(&octree, &center_nodes, &mut connectivity, data, template);
                    connectivity.len() - before
                })
                .collect();
            (0..21).for_each(|k| fired[k] += counts[k]);
            for skip in 0..21 {
                if counts[skip] == 0 {
                    continue;
                }
                let mut partial: Vec<[usize; N]> =
                    connectivity[..connectivity.len() - counts.iter().sum::<usize>()].to_vec();
                let mut offset = connectivity.len() - counts.iter().sum::<usize>();
                for (k, &count) in counts.iter().enumerate() {
                    if k != skip {
                        partial.extend_from_slice(&connectivity[offset..offset + count]);
                    }
                    offset += count;
                }
                let mesh: Mesh<3> = (
                    vec![Connectivity::Hexahedral(partial.into())],
                    coordinates.clone(),
                )
                    .into();
                if verify_dual(&mesh).is_err() {
                    broke[skip] += 1;
                }
            }
            let mesh: Mesh<3> = (
                vec![Connectivity::Hexahedral(connectivity.clone().into())],
                coordinates.clone(),
            )
                .into();
            if let Err(error) = verify_dual(&mesh) {
                panic!("strong survey baseline failed: {error}");
            }
        }
    }
    println!("template: hexes fired across survey, trees broken when knocked out");
    for k in 0..21 {
        println!(
            "  transition_{:>2}: fired {:>5}, breaks {:>3} trees if removed",
            k + 1,
            fired[k],
            broke[k]
        );
    }
}

#[test]
fn survey_weak_vertex_configs() {
    use super::super::test::verify_dual;
    use crate::geometry::ntree::Dualization;
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
    const CONFIGS: [[usize; 8]; 14] = [
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
    let mut failures = Vec::new();
    for (which, config) in CONFIGS.iter().enumerate() {
        for (way, rotation) in ROTATIONS.iter().enumerate() {
            let mut depths = [0usize; 8];
            (0..8).for_each(|octant| depths[rotation[octant]] = config[octant] + 1);
            let mut octree = super::super::edge::test::weak_tree(depths, Balancing::Weak);
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
            let coordinates = mesh.coordinates();
            let scaled_jacobian = mesh
                .iter()
                .flatten()
                .map(|hex| {
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
                            let norm =
                                |x: [f64; 3]| (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
                            det / (norm(u) * norm(v) * norm(w))
                        })
                        .fold(f64::INFINITY, f64::min)
                })
                .fold(f64::INFINITY, f64::min);
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
        panic!("{} weak vertex survey failures", failures.len());
    }
}

fn weak_vertex_depths(fine: usize) -> [usize; 8] {
    let mut depths = [2; 8];
    depths[fine] = 3;
    depths[7 - fine] = 1;
    depths
}

#[test]
fn write_weak_vertex_dual() {
    use super::super::test::verify_dual;
    use crate::{
        geometry::{mesh::Output, ntree::Dualization},
        io::Write,
    };
    for fine in 0..8 {
        let mut octree =
            super::super::edge::test::weak_tree(weak_vertex_depths(fine), Balancing::Weak);
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

#[test]
fn star_fills_weak_vertex_config_only() {
    use crate::geometry::ntree::dual::Uniform;
    let hexes = |fine, balancing| {
        let octree = super::super::edge::test::weak_tree(weak_vertex_depths(fine), balancing);
        let (center_nodes, ..) = octree.initialize();
        let mut connectivity = Vec::new();
        super::star::template(&octree, &center_nodes, &mut connectivity);
        connectivity.len()
    };
    for fine in 0..8 {
        assert_eq!(
            hexes(fine, Balancing::Weak),
            1,
            "the star template should fill the weak vertex (fine orthant {fine})"
        );
        assert_eq!(
            hexes(fine, Balancing::Strong),
            0,
            "the star template fired on the strong tree (the config should be balanced away)"
        );
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

fn one_template<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    data: &[[usize; 11]],
    template: Template<T, U>,
) -> Vec<[usize; N]>
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    let mut connectivity = Vec::new();
    apply(tree, center_nodes, &mut connectivity, data, template);
    connectivity
}

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

// The single vertex rule that the classifier below justifies: at an interior
// vertex with eight distinct incident cells, emit the star hex iff the cells
// are uniform or the vertex lies on the doubled grid of the longest cell.
fn star_general(
    tree: &Octree<u16, usize>,
    center_nodes: &[usize],
    connectivity: &mut Vec<[usize; N]>,
) {
    const WIND: [usize; N] = [0, 1, 3, 2, 4, 5, 7, 6];
    let root = &tree.nodes[0];
    let lo = root.corner;
    let hi: [u16; D] = from_fn(|a| root.corner[a] + root.length);
    for node in tree.iter().filter(|node| node.is_leaf()) {
        let v: [u16; D] = from_fn(|a| node.corner[a] + node.length);
        if !(0..D).all(|a| lo[a] < v[a] && v[a] < hi[a]) {
            continue;
        }
        let cells: [usize; N] = from_fn(|d| find_leaf_octant(tree, &v, d));
        let mut distinct = cells.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() != N {
            continue;
        }
        let shortest: usize = cells
            .iter()
            .map(|&cell| tree.nodes[cell].length.into())
            .min()
            .unwrap();
        let longest: usize = cells
            .iter()
            .map(|&cell| tree.nodes[cell].length.into())
            .max()
            .unwrap();
        if longest == shortest || (0..D).all(|a| (v[a] as usize).is_multiple_of(2 * longest)) {
            connectivity.push(from_fn(|k| center_nodes[cells[WIND[k]]]));
        }
    }
}

// The entire vertex phase (uniform transitions 1-4, the 21 strong templates,
// and the weak star) emits exactly the same hexes as the single star rule,
// on every strong and weak survey tree.
#[test]
fn star_rule_matches_vertex_phase() {
    use crate::geometry::ntree::dual::{NodeMap, Uniform};
    use std::collections::BTreeMap;
    let multiset = |hexes: &[[usize; N]]| -> BTreeMap<[usize; N], usize> {
        let mut set = BTreeMap::new();
        for hex in hexes {
            let mut sorted = *hex;
            sorted.sort_unstable();
            *set.entry(sorted).or_insert(0) += 1;
        }
        set
    };
    let mut cases: Vec<(Balancing, [usize; 8])> = Vec::new();
    for config in STRONG_CONFIGS.iter() {
        for rotation in ROTATIONS.iter() {
            let mut depths = [0usize; 8];
            (0..8).for_each(|octant| depths[rotation[octant]] = config[octant] + 1);
            cases.push((Balancing::Strong, depths));
        }
    }
    for config in WEAK_CONFIGS.iter() {
        for rotation in ROTATIONS.iter() {
            let mut depths = [0usize; 8];
            (0..8).for_each(|octant| depths[rotation[octant]] = config[octant] + 1);
            cases.push((Balancing::Weak, depths));
        }
    }
    for (balancing, depths) in cases {
        let octree = super::super::edge::test::weak_tree(depths, balancing);
        let (center_nodes, mut coordinates, mut node_index, mut connectivity) = octree.initialize();
        let mut nodes_map = NodeMap::new();
        super::super::face::face_transition(
            &octree,
            &center_nodes,
            &mut coordinates,
            &mut connectivity,
            &mut node_index,
            &mut nodes_map,
        );
        super::super::edge::edge_transitions(
            &octree,
            &center_nodes,
            &mut coordinates,
            &mut connectivity,
            &mut node_index,
            &mut nodes_map,
            balancing,
        );
        let base = connectivity.len();
        octree.uniform_transitions(&center_nodes, &mut connectivity);
        super::vertex_transitions(&octree, &center_nodes, &mut connectivity, balancing);
        let old = multiset(&connectivity[base..]);
        connectivity.truncate(base);
        star_general(&octree, &center_nodes, &mut connectivity);
        let new = multiset(&connectivity[base..]);
        assert_eq!(
            old, new,
            "vertex phase differs from the generic star rule for depths {depths:?}"
        );
    }
}

// Ground truth for the star rule's guard: classify every interior vertex
// with eight distinct incident cells across the strong survey trees by size
// ratio, tube axes, and parity on the doubled grid of the longest cell, and
// record whether the current pipeline filled it with the star hex. Every
// class is purely star or purely Steiner: uniform vertices are always
// stars, and mixed vertices are stars exactly when aligned on all axes
// (odd parities only ever occur along tube axes, which are Steiner-filled).
#[test]
fn classify_strong_distinct8_vertices() {
    use super::super::test::verify_dual;
    use crate::geometry::{
        mesh::{Connectivity, Mesh},
        ntree::dual::{NodeMap, Uniform},
    };
    use std::collections::{BTreeMap, HashSet};
    let mut trees: Vec<(String, Octree<u16, usize>)> = Vec::new();
    for (c, config) in STRONG_CONFIGS.iter().enumerate() {
        for rotation in ROTATIONS.iter() {
            let mut depths = [0usize; 8];
            (0..8).for_each(|octant| depths[rotation[octant]] = config[octant] + 1);
            trees.push((
                format!("survey {c}"),
                super::super::edge::test::weak_tree(depths, Balancing::Strong),
            ));
        }
    }
    let extras: [&[(usize, usize)]; 6] = [
        &[(0, 7)],
        &[(0, 6), (0, 7)],
        &[(0, 3), (0, 5), (0, 6), (0, 7)],
        &[(0, 1), (0, 2), (0, 4), (0, 7)],
        &[
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
        ],
        &[(0, 7), (7, 0)],
    ];
    for (e, extra) in extras.iter().enumerate() {
        trees.push((
            format!("custom {e}"),
            super::super::edge::test::weak_tree_plus([1; 8], extra, Balancing::Strong),
        ));
    }
    let mut classes: BTreeMap<String, (usize, usize, String)> = BTreeMap::new();
    for (label, octree) in trees.iter() {
        let (center_nodes, mut coordinates, mut node_index, mut connectivity) = octree.initialize();
        octree.uniform_transitions(&center_nodes, &mut connectivity);
        let mut nodes_map = NodeMap::new();
        super::super::face::face_transition(
            octree,
            &center_nodes,
            &mut coordinates,
            &mut connectivity,
            &mut node_index,
            &mut nodes_map,
        );
        super::super::edge::edge_transitions(
            octree,
            &center_nodes,
            &mut coordinates,
            &mut connectivity,
            &mut node_index,
            &mut nodes_map,
            Balancing::Strong,
        );
        super::vertex_transitions(octree, &center_nodes, &mut connectivity, Balancing::Strong);
        if label.starts_with("custom") {
            let mesh: Mesh<3> = (
                vec![Connectivity::Hexahedral(connectivity.clone().into())],
                coordinates.clone(),
            )
                .into();
            if let Err(error) = verify_dual(&mesh) {
                println!("NOTE: {label} baseline failed verification: {error}");
            }
        }
        let hexset: HashSet<[usize; N]> = connectivity
            .iter()
            .map(|hex| {
                let mut sorted = *hex;
                sorted.sort_unstable();
                sorted
            })
            .collect();
        let root = &octree.nodes[0];
        let lo = root.corner;
        let hi: [u16; D] = from_fn(|a| root.corner[a] + root.length);
        for node in octree.iter().filter(|node| node.is_leaf()) {
            let v: [u16; D] = from_fn(|a| node.corner[a] + node.length);
            if !(0..D).all(|a| lo[a] < v[a] && v[a] < hi[a]) {
                continue;
            }
            let cells: [usize; N] = from_fn(|d| find_leaf_octant(octree, &v, d));
            let mut distinct = cells.to_vec();
            distinct.sort_unstable();
            distinct.dedup();
            if distinct.len() != N {
                continue;
            }
            let sizes: [usize; N] = from_fn(|d| octree.nodes[cells[d]].length.into());
            let shortest = *sizes.iter().min().unwrap();
            let longest = *sizes.iter().max().unwrap();
            let ratio = longest / shortest;
            let mut per_axis: Vec<(bool, usize)> = (0..D)
                .map(|a| {
                    let tube = (0..N)
                        .filter(|o| o & (1 << a) == 0)
                        .all(|o| sizes[o] == sizes[o | (1 << a)]);
                    assert_eq!((v[a] as usize) % longest, 0, "{label} at {v:?}");
                    let parity = ((v[a] as usize) / longest) % 2;
                    (tube, parity)
                })
                .collect();
            per_axis.sort();
            let mut sig: [usize; N] = from_fn(|d| sizes[d] / shortest);
            sig.sort_unstable();
            let mut sorted_cells: [usize; N] = from_fn(|k| center_nodes[cells[k]]);
            sorted_cells.sort_unstable();
            let star = hexset.contains(&sorted_cells);
            let key = format!("ratio {ratio} sizes {sig:?} (tube, parity) {per_axis:?}");
            let entry = classes
                .entry(key)
                .or_insert((0, 0, format!("{label} at {v:?}")));
            if star { entry.0 += 1 } else { entry.1 += 1 }
        }
    }
    let mut mixed = Vec::new();
    for (key, (star, other, example)) in classes.iter() {
        println!("{key}: star {star:>6}, other {other:>6}   e.g. {example}");
        if *star > 0 && *other > 0 {
            mixed.push(key.clone());
        }
        let aligned = !key.contains(", 1)");
        let uniform = key.starts_with("ratio 1");
        if uniform || aligned {
            assert_eq!(*other, 0, "aligned or uniform class not star-filled: {key}");
        } else {
            assert_eq!(*star, 0, "unaligned mixed class star-filled: {key}");
        }
    }
    assert!(mixed.is_empty(), "ambiguous classes: {mixed:?}");
}

pub(crate) fn transition_21_only<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
) -> Vec<[usize; N]>
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    one_template(
        tree,
        center_nodes,
        &transition_21::DATA,
        transition_21::template,
    )
}
