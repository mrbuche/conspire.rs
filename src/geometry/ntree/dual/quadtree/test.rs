use super::{
    D, N, vertex_transition_1, vertex_transition_2, vertex_transition_3, vertex_transition_4,
    vertex_transition_5,
};
use crate::geometry::{
    mesh::Mesh,
    ntree::{
        Balance, Dualization, Quadtree,
        balance::Balancing,
        dual::{Star, Uniform},
        node::{Kind, Node},
        pair::Pairing,
        rescale::Rescaling,
    },
};
use std::collections::{HashMap, HashSet};

fn min_scaled_jacobian(mesh: &Mesh<D>) -> f64 {
    let coordinates = mesh.coordinates();
    mesh.iter()
        .flatten()
        .map(|quad| {
            (0..N)
                .map(|k| {
                    let e = |j: usize| {
                        std::array::from_fn::<f64, D, _>(|i| {
                            coordinates[quad[j]][i] - coordinates[quad[k]][i]
                        })
                    };
                    let u = e((k + 1) % N);
                    let v = e((k + N - 1) % N);
                    let det = u[0] * v[1] - u[1] * v[0];
                    let norm = |x: [f64; D]| (x[0] * x[0] + x[1] * x[1]).sqrt();
                    det / (norm(u) * norm(v))
                })
                .fold(f64::INFINITY, f64::min)
        })
        .fold(f64::INFINITY, f64::min)
}

pub(crate) fn verify_dual(mesh: &Mesh<D>) -> Result<(), String> {
    let coordinates = mesh.coordinates();
    for (e, element) in mesh.iter().flatten().enumerate() {
        let mut distinct = element.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() != N {
            return Err(format!("quad {e} has repeated nodes: {element:?}"));
        }
        let area2: f64 = (0..N)
            .map(|k| {
                let p = &coordinates[element[k]];
                let q = &coordinates[element[(k + 1) % N]];
                p[0] * q[1] - q[0] * p[1]
            })
            .sum();
        if area2 <= 1e-9 {
            return Err(format!(
                "quad {e} not positively oriented (2A={area2}): {element:?}"
            ));
        }
    }
    let mut edges: HashMap<[usize; 2], usize> = HashMap::new();
    for element in mesh.iter().flatten() {
        for k in 0..N {
            let mut edge = [element[k], element[(k + 1) % N]];
            edge.sort_unstable();
            *edges.entry(edge).or_insert(0) += 1;
        }
    }
    if let Some((edge, count)) = edges.iter().find(|(_, count)| **count > 2) {
        return Err(format!("non-conformal: edge {edge:?} shared {count} times"));
    }
    // The boundary must be a single closed loop: an unfilled interior
    // void adds a second loop, which the manifold check cannot detect.
    let boundary: Vec<[usize; 2]> = edges
        .iter()
        .filter(|(_, count)| **count == 1)
        .map(|(edge, _)| *edge)
        .collect();
    let mut degree: HashMap<usize, usize> = HashMap::new();
    for edge in &boundary {
        *degree.entry(edge[0]).or_insert(0) += 1;
        *degree.entry(edge[1]).or_insert(0) += 1;
    }
    if let Some((vertex, count)) = degree.iter().find(|(_, count)| **count != 2) {
        return Err(format!(
            "boundary not a closed manifold: vertex {vertex} borders {count} boundary edges"
        ));
    }
    let vertices: HashSet<usize> = degree.keys().copied().collect();
    let mut neighbors: HashMap<usize, Vec<usize>> = HashMap::new();
    for edge in &boundary {
        neighbors.entry(edge[0]).or_default().push(edge[1]);
        neighbors.entry(edge[1]).or_default().push(edge[0]);
    }
    let mut reached: HashSet<usize> = HashSet::new();
    let mut queue = vec![*vertices.iter().next().ok_or("boundary is empty")?];
    reached.insert(queue[0]);
    while let Some(vertex) = queue.pop() {
        for &next in neighbors.get(&vertex).into_iter().flatten() {
            if reached.insert(next) {
                queue.push(next);
            }
        }
    }
    if reached.len() != vertices.len() {
        return Err(format!(
            "boundary is disconnected ({} of {} vertices reached; unfilled interior void)",
            reached.len(),
            vertices.len()
        ));
    }
    Ok(())
}

fn fuzz_tree(seed: u64, balancing: Balancing) -> Quadtree<u16, usize> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut rand = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let mut quadtree = Quadtree::<u16, usize> {
        balanced: Balancing::None,
        nodes: vec![Node {
            corner: [0, 0],
            length: 32,
            facets: [None; 4],
            kind: Kind::Leaf,
            value: None,
        }],
        paired: Pairing::None,
        rescale: Rescaling {
            center: [16.0; D],
            cell: 1.0,
            half: 16.0,
        },
    };
    quadtree.subdivide(0).unwrap();
    for _ in 0..40 {
        let leaves: Vec<usize> = quadtree
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
        quadtree.subdivide(pick).unwrap();
    }
    quadtree.equilibrate(balancing, Pairing::Regular).unwrap();
    quadtree
}

fn fuzz_duals(balancing: Balancing) {
    let mut failures = Vec::new();
    for seed in 0..200u64 {
        let mut quadtree = fuzz_tree(seed, balancing);
        let mesh = quadtree.dualize();
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
fn fuzz_strong_duals() {
    fuzz_duals(Balancing::Strong)
}

#[test]
fn fuzz_weak_duals() {
    fuzz_duals(Balancing::Weak)
}

fn old_vertex_phase(quadtree: &Quadtree<u16, usize>, center_nodes: &[usize]) -> Vec<[usize; N]> {
    let mut connectivity = Vec::new();
    quadtree.uniform_transitions(center_nodes, &mut connectivity);
    vertex_transition_1(quadtree, center_nodes, &mut connectivity);
    vertex_transition_2(quadtree, center_nodes, &mut connectivity);
    vertex_transition_3(quadtree, center_nodes, &mut connectivity);
    vertex_transition_4(quadtree, center_nodes, &mut connectivity);
    if matches!(quadtree.balanced, Balancing::Weak) {
        vertex_transition_5(quadtree, center_nodes, &mut connectivity);
    }
    connectivity
}

fn multiset(connectivity: &[[usize; N]]) -> HashMap<[usize; N], usize> {
    let mut counts = HashMap::new();
    connectivity.iter().for_each(|quad| {
        let mut sorted = *quad;
        sorted.sort_unstable();
        *counts.entry(sorted).or_insert(0) += 1;
    });
    counts
}

fn star_matches_old_vertex_phase(balancing: Balancing) {
    let mut failures = Vec::new();
    for seed in 0..200u64 {
        let quadtree = fuzz_tree(seed, balancing);
        let (center_nodes, ..) = quadtree.initialize();
        let old = multiset(&old_vertex_phase(&quadtree, &center_nodes));
        let mut connectivity = Vec::new();
        quadtree.star(&center_nodes, &mut connectivity);
        let new = multiset(&connectivity);
        if old != new {
            let missing: Vec<_> = old.iter().filter(|(quad, _)| !new.contains_key(*quad)).collect();
            let extra: Vec<_> = new.iter().filter(|(quad, _)| !old.contains_key(*quad)).collect();
            let duplicated: Vec<_> = old.iter().filter(|(_, count)| **count > 1).collect();
            failures.push(format!(
                "seed {seed}: old {} vs new {} quads; missing from new: {missing:?}; extra in new: {extra:?}; duplicated in old: {duplicated:?}",
                old.values().sum::<usize>(),
                new.values().sum::<usize>(),
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
fn star_matches_old_vertex_phase_strong() {
    star_matches_old_vertex_phase(Balancing::Strong)
}

#[test]
fn star_matches_old_vertex_phase_weak() {
    star_matches_old_vertex_phase(Balancing::Weak)
}
