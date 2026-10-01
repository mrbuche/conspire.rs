use super::{Faces, interior_radius};
use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh},
    },
    math::{FxHashSet, Quantity},
    units::Length,
};
use std::{array::from_fn, collections::HashMap};

const TOLERANCE: f64 = 1e-9;

fn point<const D: usize>(mesh: &Mesh<D>, node: usize) -> [f64; D] {
    from_fn(|k| mesh.coordinates()[node][k].value())
}

fn length(value: f64) -> Quantity<Length> {
    Quantity::new(value)
}

fn square(n: usize, slit: bool) -> Mesh<2> {
    let h = 1.0 / n as f64;
    let mut duplicate = HashMap::new();
    let mut points: Vec<[f64; 2]> = (0..=n)
        .flat_map(|j| (0..=n).map(move |i| [i as f64 * h, j as f64 * h]))
        .collect();
    if slit {
        for i in n / 2 + 1..=n {
            duplicate.insert(i, points.len());
            points.push([i as f64 * h, 0.5]);
        }
    }
    let corner = |i: usize, j: usize, upper: bool| {
        if upper && j == n / 2 {
            duplicate.get(&i).copied().unwrap_or(j * (n + 1) + i)
        } else {
            j * (n + 1) + i
        }
    };
    let triangles: Vec<[usize; 3]> = (0..n)
        .flat_map(|j| (0..n).map(move |i| (i, j)))
        .flat_map(|(i, j)| {
            let upper = j >= n / 2;
            let [a, b, c, d] = [
                corner(i, j, upper),
                corner(i + 1, j, upper),
                corner(i + 1, j + 1, upper),
                corner(i, j + 1, upper),
            ];
            [[a, b, c], [a, c, d]]
        })
        .collect();
    let coordinates: Coordinates<2> = points.iter().map(|&p| Coordinate::from(p)).collect();
    Mesh::from((
        vec![Connectivity::Triangular(triangles.into())],
        coordinates,
    ))
}

fn assert_reproduces<const D: usize>(mesh: &Mesh<D>, basis: &super::Basis, degree: usize) {
    let mut nodes: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    basis.seeds.iter().enumerate().for_each(|(index, _)| {
        basis.values[index]
            .iter()
            .for_each(|&(node, value)| nodes.entry(node).or_default().push((index, value)))
    });
    assert!(!nodes.is_empty());
    for (node, entries) in nodes {
        let x = point(mesh, node);
        let total: f64 = entries.iter().map(|&(_, v)| v).sum();
        assert!(
            (total - 1.0).abs() < TOLERANCE,
            "sum {total} at node {node}"
        );
        if degree >= 1 {
            for (k, xk) in x.iter().enumerate() {
                let first: f64 = entries
                    .iter()
                    .map(|&(i, v)| v * point(mesh, basis.seeds[i])[k])
                    .sum();
                assert!((first - xk).abs() < TOLERANCE, "linear at node {node}");
            }
        }
        if degree >= 2 {
            let second: f64 = entries
                .iter()
                .map(|&(i, v)| v * point(mesh, basis.seeds[i])[0].powi(2))
                .sum();
            assert!(
                (second - x[0].powi(2)).abs() < TOLERANCE,
                "quadratic at {node}"
            );
        }
    }
}

#[test]
fn reproduces_on_a_square() {
    let mesh = square(40, false);
    let h = 0.2;
    let seeds = mesh.sample(length(h), 3);
    let linear = mesh
        .reproducing_basis(&seeds, length(2.6 * h), 1, 1)
        .unwrap();
    assert_reproduces(&mesh, &linear, 1);
    let quadratic = mesh
        .reproducing_basis(&seeds, length(3.2 * h), 2, 1)
        .unwrap();
    assert_reproduces(&mesh, &quadratic, 2);
}

#[test]
fn covers_every_node_and_stays_in_the_patch() {
    let mesh = square(40, false);
    let seeds = mesh.sample(length(0.2), 5);
    let radius = length(0.2 * 2.6);
    let basis = mesh.reproducing_basis(&seeds, radius, 1, 1).unwrap();
    let patches = mesh.patches(&seeds, radius);
    let mut covered = FxHashSet::default();
    for (index, patch) in patches.iter().enumerate() {
        for &(node, _) in &basis.values[index] {
            assert!(patch.nodes.contains(&node), "value outside the patch");
            covered.insert(node);
        }
    }
    assert_eq!(covered.len(), mesh.number_of_nodes());
}

#[test]
fn slit_is_respected() {
    let mesh = square(40, true);
    let h = 0.15;
    let seeds = mesh.sample(length(h), 2);
    let basis = mesh
        .reproducing_basis(&seeds, length(2.8 * h), 1, 1)
        .unwrap();
    assert_reproduces(&mesh, &basis, 1);
    let reach = 2.8 * h;
    let tip = [0.5, 0.5];
    let duplicates = 41 * 41;
    let mut checked = 0;
    for (index, &seed) in seeds.iter().enumerate() {
        let [x, y] = point(&mesh, seed);
        if y < 0.5 && (x - tip[0]).hypot(y - tip[1]) > reach && x > tip[0] {
            checked += 1;
            assert!(
                basis.values[index]
                    .iter()
                    .all(|&(node, _)| node < duplicates && point(&mesh, node)[1] <= 0.5 + 1e-12),
                "a seed below the slit reaches above it"
            );
        }
    }
    assert!(checked > 0, "no seed was checked");
}

#[test]
fn interior_radius_ignores_the_domain_boundary() {
    let mesh = square(40, false);
    let exterior: FxHashSet<Vec<usize>> = mesh
        .exterior_faces()
        .into_iter()
        .map(|mut face| {
            face.sort_unstable();
            face
        })
        .collect();
    let elements: Vec<_> = mesh
        .iter()
        .flat_map(|block| block.iter().map(move |element| (block, element)))
        .collect();
    let faces = Faces::new(&elements, &exterior);
    let radius = 0.3;
    let radius_at = |target: [f64; 2]| {
        let seed = (0..mesh.number_of_nodes())
            .min_by(|&a, &b| {
                let d = |n: usize| {
                    let p = point(&mesh, n);
                    (p[0] - target[0]).hypot(p[1] - target[1])
                };
                d(a).total_cmp(&d(b))
            })
            .unwrap();
        let patch = mesh.patch(seed, length(radius));
        let distances: Vec<(usize, f64)> = mesh
            .geodesic_distances_over(seed, &patch.elements)
            .unwrap()
            .into_iter()
            .map(|(node, d)| (node, d.value_as::<Length>()))
            .collect();
        interior_radius(
            &faces,
            &mut vec![0; faces.nodes.len()],
            &patch,
            &distances,
            radius,
        )
    };
    let interior = radius_at([0.5, 0.5]);
    assert!(interior > 0.8 * radius && interior <= radius, "{interior}");
    let near_edge = radius_at([0.1, 0.5]);
    assert!(
        near_edge > 0.8 * radius,
        "the domain boundary shrank the radius to {near_edge}"
    );
}

#[test]
fn vanishes_where_the_patch_is_cut() {
    let mesh = square(40, false);
    let seeds = mesh.sample(length(0.2), 4);
    let radius = length(0.2 * 2.6);
    let basis = mesh.reproducing_basis(&seeds, radius, 1, 1).unwrap();
    let patches = mesh.patches(&seeds, radius);
    let boundary: FxHashSet<usize> = mesh.exterior_faces().into_iter().flatten().collect();
    let mut checked = 0;
    for (index, patch) in patches.iter().enumerate() {
        for &node in &patch.nodes {
            let cut = mesh.node_node_connectivity()[node]
                .iter()
                .any(|next| !patch.nodes.contains(next));
            if cut && !boundary.contains(&node) {
                checked += 1;
                assert!(
                    basis.values[index].iter().all(|&(n, _)| n != node),
                    "a basis function is nonzero on the cut edge of its patch"
                );
            }
        }
    }
    assert!(checked > 0);
}

#[test]
fn errors() {
    let mesh = square(20, false);
    let far = mesh.sample(length(0.5), 1);
    assert_eq!(
        mesh.reproducing_basis(&far, length(0.1), 1, 1).unwrap_err(),
        "seeds do not cover the mesh"
    );
    let seeds = mesh.sample(length(0.2), 1);
    assert_eq!(
        mesh.reproducing_basis(&seeds, length(0.21), 1, 1)
            .unwrap_err(),
        "seeds do not cover the mesh"
    );
    assert!(
        mesh.reproducing_basis(&seeds, length(10.0), 12, 1).is_err(),
        "an impossible degree must be an error"
    );
}

#[test]
fn threads_give_the_same_basis() {
    for (slit, spacing, reach, seed) in [(false, 0.2, 2.6, 4), (true, 0.15, 2.8, 2)] {
        let mesh = square(40, slit);
        let seeds = mesh.sample(length(spacing), seed);
        let radius = length(spacing * reach);
        let serial = mesh.reproducing_basis(&seeds, radius, 1, 1).unwrap();
        for threads in [1, 2, 3, usize::MAX] {
            assert_eq!(
                mesh.reproducing_basis(&seeds, radius, 1, threads).unwrap(),
                serial,
                "{threads} threads, slit {slit}"
            );
        }
    }
}

#[test]
fn threads_report_the_same_errors() {
    let mesh = square(20, false);
    let far = mesh.sample(length(0.5), 1);
    assert_eq!(
        mesh.reproducing_basis(&far, length(0.1), 1, usize::MAX)
            .unwrap_err(),
        "seeds do not cover the mesh"
    );
}
