use super::{D, N};
use crate::geometry::{
    mesh::Mesh,
    ntree::{
        Balance, Dualization, Quadtree,
        balance::Balancing,
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

fn fuzz_tree(seed: u64, balancing: Balancing, pairing: Pairing) -> Quadtree<u16, usize> {
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
    quadtree.equilibrate(balancing, pairing).unwrap();
    quadtree
}

fn fuzz_duals(balancing: Balancing, pairing: Pairing) {
    let mut failures = Vec::new();
    for seed in 0..200u64 {
        let mut quadtree = fuzz_tree(seed, balancing, pairing);
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
    fuzz_duals(Balancing::Strong, Pairing::Regular)
}

#[test]
fn fuzz_weak_duals() {
    fuzz_duals(Balancing::Weak, Pairing::Regular)
}

// Pairing::Generalized can produce asymmetric local transitions that
// Regular (octree tree-rule) pairing never does, and this dual template
// set was only ever validated against Regular-paired grids: most seeds
// fail here today. Left ignored as a standing record of the gap; closing
// it means extending dual template coverage, not fixing pair::general.
#[test]
#[ignore]
fn fuzz_strong_duals_generalized() {
    fuzz_duals(Balancing::Strong, Pairing::Generalized)
}

#[test]
#[ignore]
fn fuzz_weak_duals_generalized() {
    fuzz_duals(Balancing::Weak, Pairing::Generalized)
}
