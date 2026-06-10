use crate::{
    geometry::mesh::{Connectivity, Mesh, Verdict},
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
fn untangles_an_inverted_interior_node() {
    let (connectivity, mut coordinates) = grid();
    coordinates[13] = [1.0, 1.0, 2.5];
    let mut mesh = mesh(connectivity, coordinates);
    assert!(minimum_scaled(&mesh) < 0.0);
    mesh.untangle(50, 0.1, None);
    assert!(minimum_scaled(&mesh) > 0.0);
}

#[test]
fn leaves_a_valid_mesh_alone() {
    let (connectivity, coordinates) = grid();
    let mut mesh = mesh(connectivity, coordinates);
    let before: Vec<_> = mesh.coordinates().iter().cloned().collect();
    mesh.untangle(10, 0.1, None);
    mesh.coordinates()
        .iter()
        .zip(before)
        .for_each(|(now, was)| {
            assert_eq!(now, &was);
        });
}
