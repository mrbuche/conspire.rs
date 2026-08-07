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
    let used: HashSet<usize> = mesh.iter().flatten().flatten().copied().collect();
    let faces = mesh.iter().flatten().count();
    let euler = used.len() as isize - edges.len() as isize + faces as isize;
    if euler != 1 {
        return Err(format!(
            "euler characteristic {euler}, not a disc ({} vertices, {} edges, {faces} faces)",
            used.len(),
            edges.len()
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
        pairing_vertices: Default::default(),
        rescale: Rescaling {
            center: [16.0; D],
            cell: 1.0,
            half: 16.0,
        },
    };
    quadtree.subdivide(0).unwrap();
    for _ in 0..60 {
        let leaves: Vec<usize> = quadtree
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.is_leaf() && node.length >= 2)
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
    for seed in 0..100u64 {
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
    fuzz_duals(Balancing::Strong(1), Pairing::Regular)
}

#[test]
fn fuzz_weak_duals() {
    fuzz_duals(Balancing::Weak(1), Pairing::Regular)
}

#[test]
fn fuzz_strong_duals_generalized() {
    fuzz_duals(Balancing::Strong(1), Pairing::Generalized)
}

#[test]
fn fuzz_weak_duals_generalized() {
    fuzz_duals(Balancing::Weak(1), Pairing::Generalized)
}

// The transition takes a half of a cluster facet - one coarse leaf and the two fine cells behind
// it - only when the half is whole, and treats a facet as truncated only when every absent half
// is provably outside the domain. A half absent for any other reason (the outside cell refined,
// the inside cells refined deeper) means the transition belongs to a different cluster, and
// guessing a template there would double-cover it. This asserts that mixed facets - one half
// present, the other absent while still inside the domain - never arise, so the classification
// the template relies on is total rather than merely sound.
#[test]
fn every_cluster_facet_is_classifiable() {
    use crate::geometry::ntree::dual::leaf_containing;
    for pairing in [Pairing::Regular, Pairing::Generalized] {
        let mut mixed = Vec::new();
        for seed in 0..100u64 {
            let tree = fuzz_tree(seed, Balancing::Weak(1), pairing);
            let root = &tree.nodes[0];
            let low: [i64; D] = std::array::from_fn(|a| root.corner[a] as i64);
            let high: [i64; D] = std::array::from_fn(|a| low[a] + root.length as i64);
            let cell_at = |corner: [i64; D], length: i64| -> Option<usize> {
                if (0..D).any(|a| corner[a] < low[a] || corner[a] + length > high[a]) {
                    return None;
                }
                let point = std::array::from_fn(|a| corner[a] as usize);
                let index = leaf_containing(&tree, &point);
                let node = &tree.nodes[index];
                (length as usize == node.length as usize
                    && (0..D).all(|a| point[a] == node.corner[a] as usize))
                .then_some(index)
            };
            for &(cluster, length) in tree.pairing_vertices.iter() {
                let center: [i64; D] = std::array::from_fn(|a| cluster[a] as i64);
                let (coarse, fine) = (length as i64, length as i64 / 2);
                for facet in 0..4 {
                    let (axis, side) = (facet >> 1, facet & 1);
                    let tangent = 1 - axis;
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
                    let base = center[tangent] - coarse;
                    let at = |along, across| {
                        let mut c = [0; D];
                        c[axis] = along;
                        c[tangent] = across;
                        c
                    };
                    let present = |h: i64| {
                        cell_at(at(outside, base + h * coarse), coarse).is_some()
                            && cell_at(at(inside, base + 2 * h * fine), fine).is_some()
                            && cell_at(at(inside, base + (2 * h + 1) * fine), fine).is_some()
                    };
                    let outside_domain = |h: i64| {
                        base + h * coarse < low[tangent] || base + (h + 1) * coarse > high[tangent]
                    };
                    let any = (0..2).any(present);
                    let stray = (0..2).any(|h| !present(h) && !outside_domain(h));
                    if any && stray {
                        mixed.push(format!("seed {seed} cluster {cluster:?} facet {facet}"));
                    }
                }
            }
        }
        assert!(
            mixed.is_empty(),
            "{} facets mix a present half with an absent in-domain one:\n{}",
            mixed.len(),
            mixed.join("\n")
        );
    }
}
