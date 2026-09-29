use crate::{
    geometry::{
        Coordinate, Coordinates,
        grid::Voxels,
        mesh::{Connectivity, Mesh, Patch},
    },
    math::Quantity,
};
use std::{array::from_fn, collections::HashMap};

fn point<const D: usize>(mesh: &Mesh<D>, node: usize) -> [f64; D] {
    from_fn(|k| mesh.coordinates()[node][k].value())
}

fn distance<const D: usize>(a: [f64; D], b: [f64; D]) -> f64 {
    (0..D).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt()
}

fn nearest<const D: usize>(mesh: &Mesh<D>, target: [f64; D]) -> usize {
    (0..mesh.number_of_nodes())
        .filter(|&node| !mesh.node_element_connectivity()[node].is_empty())
        .min_by(|&a, &b| {
            distance(point(mesh, a), target).total_cmp(&distance(point(mesh, b), target))
        })
        .unwrap()
}

fn ball<const D: usize>(mesh: &Mesh<D>, seed: usize, radius: f64) -> Vec<usize> {
    mesh.iter()
        .flat_map(|block| {
            block
                .iter()
                .map(move |element| block.element_nodes(element))
        })
        .enumerate()
        .filter(|(_, nodes)| {
            nodes
                .iter()
                .all(|&node| distance(point(mesh, node), point(mesh, seed)) <= radius)
        })
        .map(|(element, _)| element)
        .collect()
}

fn assert_is_the_ball<const D: usize>(mesh: &Mesh<D>, seed: usize, radius: f64, patch: &Patch) {
    let reach = ball(mesh, seed, radius);
    assert!(
        patch.elements.iter().all(|e| reach.contains(e)),
        "patch reaches outside the ball"
    );
    let left_out: Vec<usize> = reach
        .iter()
        .copied()
        .filter(|e| !patch.elements.contains(e))
        .collect();
    let elements: Vec<Vec<usize>> = mesh
        .iter()
        .flat_map(|block| {
            block
                .iter()
                .map(move |element| block.element_nodes(element))
        })
        .collect();
    left_out.iter().for_each(|&e| {
        assert!(
            elements[e].iter().all(|node| !patch.nodes.contains(node)),
            "a ball element touching the patch was left out"
        )
    });
    assert!(
        (left_out.len() as f64) < 0.05 * reach.len() as f64,
        "{} of {} ball elements left out",
        left_out.len(),
        reach.len()
    );
}

fn square(n: usize, slit: bool, split: bool) -> Mesh<2> {
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
    let (mut first, mut second) = (Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..n {
            let upper = j >= n / 2;
            let [a, b, c, d] = [
                corner(i, j, upper),
                corner(i + 1, j, upper),
                corner(i + 1, j + 1, upper),
                corner(i, j + 1, upper),
            ];
            first.push([a, b, c]);
            second.push([a, c, d]);
        }
    }
    let connectivities = if split {
        vec![
            Connectivity::Triangular(first.into()),
            Connectivity::Triangular(second.into()),
        ]
    } else {
        first.extend(second);
        vec![Connectivity::Triangular(first.into())]
    };
    let coordinates: Coordinates<2> = points.iter().map(|&p| Coordinate::from(p)).collect();
    Mesh::from((connectivities, coordinates))
}

fn tetrahedra(n: usize, keep: impl Fn([f64; 3]) -> bool) -> Mesh<3> {
    let h = 1.0 / n as f64;
    let node = |i: usize, j: usize, k: usize| (k * (n + 1) + j) * (n + 1) + i;
    let coordinates: Coordinates<3> = (0..=n)
        .flat_map(|k| {
            (0..=n).flat_map(move |j| {
                (0..=n).map(move |i| Coordinate::from([i as f64 * h, j as f64 * h, k as f64 * h]))
            })
        })
        .collect();
    let orders = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let tetrahedra: Vec<[usize; 4]> = (0..n)
        .flat_map(|k| (0..n).flat_map(move |j| (0..n).map(move |i| [i, j, k])))
        .flat_map(|cell| orders.map(|order| (cell, order)))
        .map(|(cell, order)| {
            let mut corner = cell;
            let mut nodes = [node(corner[0], corner[1], corner[2]); 4];
            let mut points = [corner.map(|c| c as f64 * h); 4];
            for (step, axis) in order.into_iter().enumerate() {
                corner[axis] += 1;
                nodes[step + 1] = node(corner[0], corner[1], corner[2]);
                points[step + 1] = corner.map(|c| c as f64 * h);
            }
            (nodes, points)
        })
        .filter(|(_, points)| keep(from_fn(|k| points.iter().map(|p| p[k]).sum::<f64>() / 4.0)))
        .map(|(nodes, _)| nodes)
        .collect();
    Mesh::from((
        vec![Connectivity::Tetrahedral(tetrahedra.into())],
        coordinates,
    ))
}

fn radius(value: f64) -> Quantity<crate::units::Length> {
    Quantity::new(value)
}

#[test]
fn convex_square_is_the_ball() {
    let mesh = square(20, false, false);
    let seed = nearest(&mesh, [0.5, 0.5]);
    let patch = mesh.patch(seed, radius(0.26));
    assert!(!patch.elements.is_empty());
    assert_is_the_ball(&mesh, seed, 0.26, &patch);
    assert!(patch.nodes.contains(&seed));
    assert!(patch.nodes.windows(2).all(|w| w[0] < w[1]));
}

#[test]
fn patch_across_element_blocks() {
    let joined = square(20, false, false);
    let split = square(20, false, true);
    let seed = nearest(&split, [0.5, 0.5]);
    let patch = split.patch(seed, radius(0.26));
    assert_is_the_ball(&split, seed, 0.26, &patch);
    let first = patch.elements.iter().filter(|&&e| e < 400).count();
    assert!(first > 0 && first < patch.elements.len(), "one block only");
    assert_eq!(
        patch.elements.len(),
        joined.patch(seed, radius(0.26)).elements.len()
    );
}

#[test]
fn slit_is_not_crossed() {
    let mesh = square(40, true, false);
    let seed = nearest(&mesh, [0.75, 0.45]);
    let patch = mesh.patch(seed, radius(0.152));
    let reach = ball(&mesh, seed, 0.152);
    assert!(
        reach.len() > patch.elements.len(),
        "the ball must reach across the slit"
    );
    assert!(
        patch
            .nodes
            .iter()
            .all(|&node| point(&mesh, node)[1] <= 0.5 + 1e-12),
        "patch crossed the slit"
    );
}

#[test]
fn tetrahedral_box_is_the_ball() {
    let mesh = tetrahedra(8, |_| true);
    let seed = nearest(&mesh, [0.5, 0.5, 0.5]);
    let patch = mesh.patch(seed, radius(0.3));
    assert!(!patch.elements.is_empty());
    assert_is_the_ball(&mesh, seed, 0.3, &patch);
}

#[test]
fn wall_is_not_crossed() {
    let mesh = tetrahedra(16, |c| !((0.4..0.6).contains(&c[0]) && c[1] < 0.7));
    let seed = nearest(&mesh, [0.3, 0.2, 0.5]);
    let patch = mesh.patch(seed, radius(0.4));
    let reach = ball(&mesh, seed, 0.4);
    assert!(reach.len() > patch.elements.len());
    assert!(
        patch.nodes.iter().all(|&node| point(&mesh, node)[0] <= 0.5),
        "patch crossed the wall"
    );
}

#[test]
fn hexahedral_cube_is_the_ball() {
    let mesh = Mesh::from_voxels(Voxels::new(vec![1u8; 6 * 6 * 6], [6, 6, 6]), None);
    let seed = nearest(&mesh, [3.0, 3.0, 3.0]);
    let patch = mesh.patch(seed, radius(2.0));
    assert!(!patch.elements.is_empty());
    assert_is_the_ball(&mesh, seed, 2.0, &patch);
}

#[test]
fn many_seeds_match_single_patches() {
    let mesh = square(20, true, true);
    let seeds = [
        nearest(&mesh, [0.2, 0.2]),
        nearest(&mesh, [0.75, 0.45]),
        nearest(&mesh, [0.5, 0.9]),
    ];
    let patches = mesh.patches(&seeds, radius(0.2));
    seeds
        .iter()
        .zip(&patches)
        .for_each(|(&seed, patch)| assert_eq!(patch, &mesh.patch(seed, radius(0.2))));
}

#[test]
fn small_and_degenerate_radii() {
    let mesh = square(10, false, false);
    let seed = nearest(&mesh, [0.5, 0.5]);
    assert!(mesh.patch(seed, radius(0.0)).elements.is_empty());
    assert!(mesh.patch(seed, radius(0.05)).elements.is_empty());
    let everything = mesh.patch(seed, radius(10.0));
    assert_eq!(everything.elements.len(), mesh.number_of_elements());
    assert_eq!(everything.nodes.len(), mesh.number_of_nodes());
}

#[test]
fn seed_without_elements_has_an_empty_patch() {
    let mut coordinates: Vec<Coordinate<2>> = (0..3)
        .map(|k| Coordinate::from([k as f64, (k % 2) as f64]))
        .collect();
    coordinates.push(Coordinate::from([5.0, 5.0]));
    let mesh = Mesh::from((
        vec![Connectivity::Triangular(vec![[0usize, 1, 2]].into())],
        Coordinates::from_iter(coordinates),
    ));
    let patch = mesh.patch(3, radius(100.0));
    assert!(patch.elements.is_empty() && patch.nodes.is_empty());
}

#[test]
#[should_panic(expected = "Patch radius must not be negative.")]
fn negative_radius() {
    square(2, false, false).patch(0, radius(-1.0));
}

#[test]
#[should_panic(expected = "Patch seed must be a node of the mesh.")]
fn seed_out_of_range() {
    square(2, false, false).patch(1000, radius(1.0));
}
