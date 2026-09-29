use crate::{
    geometry::{
        Coordinate, Coordinates,
        grid::Voxels,
        mesh::{Connectivity, Mesh},
    },
    math::Quantity,
    units::Length,
};
use std::{array::from_fn, collections::HashMap, f64::consts::PI};

const N: usize = 40;

fn grid(n: usize, slit: bool, place: impl Fn(f64, f64) -> [f64; 3]) -> Mesh<3> {
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
    let coordinates: Coordinates<3> = points
        .iter()
        .map(|p| Coordinate::from(place(p[0], p[1])))
        .collect();
    Mesh::from((
        vec![Connectivity::Triangular(triangles.into())],
        coordinates,
    ))
}

fn flat(n: usize, slit: bool) -> Mesh<3> {
    grid(n, slit, |x, y| [x, y, 0.0])
}

fn point(mesh: &Mesh<3>, node: usize) -> [f64; 3] {
    from_fn(|k| mesh.coordinates()[node][k].value())
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt()
}

fn nearest(mesh: &Mesh<3>, target: [f64; 3]) -> usize {
    (0..mesh.number_of_nodes())
        .min_by(|&a, &b| {
            distance(point(mesh, a), target).total_cmp(&distance(point(mesh, b), target))
        })
        .unwrap()
}

fn values(distances: Vec<Quantity<Length>>) -> Vec<f64> {
    distances.iter().map(|d| d.value_as::<Length>()).collect()
}

fn worst_relative_error(mesh: &Mesh<3>, source: usize, distances: &[(usize, f64)]) -> f64 {
    let h = 1.0 / N as f64;
    distances
        .iter()
        .filter_map(|&(node, d)| {
            let exact = distance(point(mesh, node), point(mesh, source));
            (exact > 6.0 * h).then(|| (d - exact).abs() / exact)
        })
        .fold(0.0, f64::max)
}

#[test]
fn flat_square_matches_euclidean() {
    let mesh = flat(N, false);
    let source = nearest(&mesh, [0.5, 0.5, 0.0]);
    let distances = values(mesh.geodesic_distances(source).unwrap());
    assert_eq!(distances[source], 0.0);
    let pairs: Vec<(usize, f64)> = distances.iter().copied().enumerate().collect();
    let worst = worst_relative_error(&mesh, source, &pairs);
    assert!(worst < 0.08, "worst relative error {worst}");
}

#[test]
fn tilted_plane_matches_euclidean() {
    let (c, s) = (0.6f64.cos(), 0.6f64.sin());
    let mesh = grid(N, false, |x, y| [x + 1.0, c * y - 2.0, s * y + 3.0]);
    let source = nearest(&mesh, [1.5, c * 0.5 - 2.0, s * 0.5 + 3.0]);
    let pairs: Vec<(usize, f64)> = values(mesh.geodesic_distances(source).unwrap())
        .into_iter()
        .enumerate()
        .collect();
    let worst = worst_relative_error(&mesh, source, &pairs);
    assert!(worst < 0.08, "worst relative error {worst}");
}

#[test]
fn wraps_around_a_slit() {
    let mesh = flat(N, true);
    let source = nearest(&mesh, [0.75, 0.6, 0.0]);
    let across = nearest(&mesh, [0.75, 0.4, 0.0]);
    let d = values(mesh.geodesic_distances(source).unwrap());
    let euclidean = distance(point(&mesh, across), point(&mesh, source));
    let expected = 2.0 * 0.25f64.hypot(0.1);
    assert!(euclidean < 0.25);
    assert!(d[across] > 0.8 * expected, "{} vs {expected}", d[across]);
    assert!(d[across] < 1.2 * expected, "{} vs {expected}", d[across]);
}

#[test]
fn follows_a_curved_surface() {
    let mesh = grid(N, false, |t, z| {
        let angle = PI * t;
        [angle.cos(), angle.sin(), z]
    });
    let source = nearest(&mesh, [1.0, 0.0, 0.5]);
    let opposite = nearest(&mesh, [-1.0, 0.0, 0.5]);
    let d = values(mesh.geodesic_distances(source).unwrap());
    let chord = distance(point(&mesh, source), point(&mesh, opposite));
    assert!((chord - 2.0).abs() < 1e-9);
    let arc = PI;
    assert!(
        (d[opposite] - arc).abs() < 0.1 * arc,
        "{} vs arc {arc}",
        d[opposite]
    );
}

#[test]
fn subset_of_elements() {
    let mesh = flat(N, false);
    let source = nearest(&mesh, [0.5, 0.5, 0.0]);
    let triangles: Vec<[usize; 3]> = mesh
        .iter()
        .flat_map(|block| block.iter().map(|e| [e[0], e[1], e[2]]))
        .collect();
    let elements: Vec<usize> = (0..triangles.len())
        .filter(|&e| {
            triangles[e]
                .iter()
                .all(|&node| distance(point(&mesh, node), [0.5, 0.5, 0.0]) < 0.3)
        })
        .collect();
    let distances: Vec<(usize, f64)> = mesh
        .geodesic_distances_over(source, &elements)
        .unwrap()
        .into_iter()
        .map(|(node, d)| (node, d.value_as::<Length>()))
        .collect();
    assert!(distances.len() < mesh.number_of_nodes());
    assert!(distances.windows(2).all(|w| w[0].0 < w[1].0));
    let worst = worst_relative_error(&mesh, source, &distances);
    assert!(worst < 0.08, "worst relative error {worst}");
}

#[test]
fn unreached_nodes_are_infinitely_far() {
    let mut coordinates: Vec<Coordinate<3>> = (0..3)
        .map(|k| Coordinate::from([k as f64, (k % 2) as f64, 0.0]))
        .collect();
    coordinates.push(Coordinate::from([5.0, 5.0, 5.0]));
    let mesh = Mesh::from((
        vec![Connectivity::Triangular(vec![[0usize, 1, 2]].into())],
        Coordinates::from_iter(coordinates),
    ));
    let d = values(mesh.geodesic_distances(0).unwrap());
    assert_eq!(d[0], 0.0);
    assert!(d[1].is_finite() && d[2].is_finite());
    assert_eq!(d[3], f64::INFINITY);
}

#[test]
fn errors() {
    let mesh = flat(4, false);
    assert!(mesh.geodesic_distances_over(0, &[]).is_err());
    let far = mesh.number_of_nodes() - 1;
    assert_eq!(
        mesh.geodesic_distances_over(far, &[0]).unwrap_err(),
        "source node is not in the elements"
    );
    let hexahedron = Mesh::from_voxels(Voxels::new(vec![1u8], [1, 1, 1]), None);
    assert_eq!(
        hexahedron.geodesic_distances(0).unwrap_err(),
        "geodesic distances require a triangular or tetrahedral mesh"
    );
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
            let mut points = [[0.0; 3]; 4];
            points[0] = corner.map(|c| c as f64 * h);
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

fn box_error(n: usize) -> f64 {
    let mesh = tetrahedra(n, |_| true);
    let source = nearest(&mesh, [0.5, 0.5, 0.5]);
    let d = values(mesh.geodesic_distances(source).unwrap());
    assert_eq!(d[source], 0.0);
    (0..mesh.number_of_nodes())
        .filter_map(|node| {
            let exact = distance(point(&mesh, node), point(&mesh, source));
            (exact > 0.3).then(|| (d[node] - exact).abs() / exact)
        })
        .fold(0.0, f64::max)
}

#[test]
fn tetrahedral_box_converges_to_euclidean() {
    let (coarse, fine) = (box_error(8), box_error(16));
    assert!(fine < coarse, "{fine} not below {coarse}");
    assert!(fine < 0.15, "worst relative error {fine}");
}

fn wall_distance(n: usize) -> f64 {
    let mesh = tetrahedra(n, |c| !((0.4..0.6).contains(&c[0]) && c[1] < 0.7));
    let source = nearest(&mesh, [0.25, 0.2, 0.5]);
    let across = nearest(&mesh, [0.75, 0.2, 0.5]);
    let euclidean = distance(point(&mesh, across), point(&mesh, source));
    assert!((euclidean - 0.5).abs() < 1e-9);
    values(mesh.geodesic_distances(source).unwrap())[across]
}

#[test]
fn tetrahedra_wrap_around_a_wall() {
    let expected = 2.0 * 0.15f64.hypot(0.5) + 0.2;
    let (coarse, fine) = (wall_distance(12), wall_distance(16));
    assert!(fine < coarse, "{fine} not below {coarse}");
    assert!(fine > 0.85 * expected, "{fine} vs {expected}");
    assert!(fine < 1.2 * expected, "{fine} vs {expected}");
}
