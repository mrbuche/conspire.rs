use crate::{
    geometry::{
        Coordinate,
        mesh::{Connectivity, Mesh, Verdict},
    },
    math::Tensor,
};

fn grid() -> (Vec<[usize; 8]>, Vec<[f64; 3]>) {
    let id = |i: usize, j: usize, k: usize| i + 3 * j + 9 * k;
    let mut coordinates = Vec::new();
    for k in 0..3 {
        for j in 0..3 {
            for i in 0..3 {
                coordinates.push([i as f64, j as f64, k as f64]);
            }
        }
    }
    let mut connectivity = Vec::new();
    for k in 0..2 {
        for j in 0..2 {
            for i in 0..2 {
                connectivity.push([
                    id(i, j, k),
                    id(i + 1, j, k),
                    id(i + 1, j + 1, k),
                    id(i, j + 1, k),
                    id(i, j, k + 1),
                    id(i + 1, j, k + 1),
                    id(i + 1, j + 1, k + 1),
                    id(i, j + 1, k + 1),
                ]);
            }
        }
    }
    (connectivity, coordinates)
}

fn mesh(connectivity: Vec<[usize; 8]>, coordinates: Vec<[f64; 3]>) -> Mesh<3> {
    Mesh::from((
        vec![Connectivity::Hexahedral(connectivity.into())],
        coordinates.into(),
    ))
}

fn minimum_scaled(mesh: &Mesh<3>) -> f64 {
    mesh.minimum_scaled_jacobians()
        .into_iter()
        .flatten()
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn interior_node_recenters_without_dropping_quality() {
    let (connectivity, mut coordinates) = grid();
    coordinates[13] = [1.4, 1.3, 0.6];
    let mut mesh = mesh(connectivity, coordinates);
    let center: Coordinate<3> = [1.0, 1.0, 1.0].into();
    let before = minimum_scaled(&mesh);
    let offset = (&mesh.coordinates()[13] - &center).norm();
    mesh.smart_laplace_smooth(10, 1.0);
    assert!(minimum_scaled(&mesh) >= before);
    assert!((&mesh.coordinates()[13] - &center).norm() < offset);
}

#[test]
fn perfect_grid_does_not_move() {
    let (connectivity, coordinates) = grid();
    let mut mesh = mesh(connectivity, coordinates);
    let before: Vec<_> = mesh.coordinates().iter().cloned().collect();
    mesh.smart_laplace_smooth(5, 1.0);
    mesh.coordinates()
        .iter()
        .zip(before)
        .for_each(|(now, was)| assert!((now - &was).norm() < 1.0e-12));
}
