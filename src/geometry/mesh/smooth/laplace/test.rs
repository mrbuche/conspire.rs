use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh, differential::laplace::Weighting},
    },
    math::{
        Scalar, Tensor,
        test::{TestError, assert_eq_within_tols},
    },
};

fn tri() -> Mesh<3> {
    Mesh::from((
        vec![Connectivity::Triangular(vec![[0_usize, 1, 2]].into())],
        Coordinates::from([
            Coordinate::const_from([0.0, 0.0, 0.0]),
            Coordinate::const_from([2.0, 0.0, 0.0]),
            Coordinate::const_from([0.0, 2.0, 0.0]),
        ]),
    ))
}

fn centroid(mesh: &Mesh<3>) -> Coordinate<3> {
    mesh.coordinates().iter().sum::<Coordinate<3>>() / 3.0
}

fn spread(mesh: &Mesh<3>) -> Scalar {
    let coordinates = mesh.coordinates();
    let center = centroid(mesh);
    (0..3)
        .map(|node| {
            (0..3)
                .map(|i| (coordinates[node][i] - center[i]).powi(2))
                .sum::<Scalar>()
        })
        .sum()
}

#[test]
fn full_step_moves_each_vertex_to_neighbor_centroid() -> Result<(), TestError> {
    let mut mesh = tri();
    mesh.laplace_smooth(1, 1.0, Weighting::Uniform);
    let coordinates = mesh.coordinates();
    assert_eq_within_tols(&coordinates[0], &[1.0, 1.0, 0.0].into())?;
    assert_eq_within_tols(&coordinates[1], &[0.0, 1.0, 0.0].into())?;
    assert_eq_within_tols(&coordinates[2], &[1.0, 0.0, 0.0].into())
}

#[test]
fn zero_scale_is_identity() -> Result<(), TestError> {
    let mut mesh = tri();
    mesh.laplace_smooth(5, 0.0, Weighting::Uniform);
    let coordinates = mesh.coordinates();
    assert_eq_within_tols(&coordinates[0], &[0.0, 0.0, 0.0].into())?;
    assert_eq_within_tols(&coordinates[1], &[2.0, 0.0, 0.0].into())?;
    assert_eq_within_tols(&coordinates[2], &[0.0, 2.0, 0.0].into())
}

#[test]
fn preserves_centroid() -> Result<(), TestError> {
    let before = centroid(&tri());
    let mut mesh = tri();
    mesh.laplace_smooth(4, 0.5, Weighting::Uniform);
    assert_eq_within_tols(&before, &centroid(&mesh))
}

#[test]
fn shrinks_toward_centroid() {
    let before = spread(&tri());
    let mut mesh = tri();
    mesh.laplace_smooth(4, 0.5, Weighting::Uniform);
    assert!(spread(&mesh) < before);
}

#[test]
fn cotangent_full_step() -> Result<(), TestError> {
    let mut mesh = tri();
    mesh.laplace_smooth(1, 1.0, Weighting::Cotangent);
    let coordinates = mesh.coordinates();
    assert_eq_within_tols(&coordinates[0], &[1.0, 1.0, 0.0].into())?;
    assert_eq_within_tols(&coordinates[1], &[0.0, 0.0, 0.0].into())?;
    assert_eq_within_tols(&coordinates[2], &[0.0, 0.0, 0.0].into())
}
