use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh, differential::laplace::Weighting},
    },
    math::{
        Scalar, Tensor,
        assert::{AssertionError, assert_eq_within_tols},
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
fn full_step_moves_each_vertex_to_neighbor_centroid() -> Result<(), AssertionError> {
    let mut mesh = tri();
    mesh.laplace_smooth(1, 1.0, Weighting::Uniform, false, false);
    let coordinates = mesh.coordinates();
    assert_eq_within_tols(&coordinates[0], &[1.0, 1.0, 0.0].into())?;
    assert_eq_within_tols(&coordinates[1], &[0.0, 1.0, 0.0].into())?;
    assert_eq_within_tols(&coordinates[2], &[1.0, 0.0, 0.0].into())
}

#[test]
fn zero_scale_is_identity() -> Result<(), AssertionError> {
    let mut mesh = tri();
    mesh.laplace_smooth(5, 0.0, Weighting::Uniform, false, false);
    let coordinates = mesh.coordinates();
    assert_eq_within_tols(&coordinates[0], &[0.0, 0.0, 0.0].into())?;
    assert_eq_within_tols(&coordinates[1], &[2.0, 0.0, 0.0].into())?;
    assert_eq_within_tols(&coordinates[2], &[0.0, 2.0, 0.0].into())
}

#[test]
fn preserves_centroid() -> Result<(), AssertionError> {
    let before = centroid(&tri());
    let mut mesh = tri();
    mesh.laplace_smooth(4, 0.5, Weighting::Uniform, false, false);
    assert_eq_within_tols(&before, &centroid(&mesh))
}

#[test]
fn shrinks_toward_centroid() {
    let before = spread(&tri());
    let mut mesh = tri();
    mesh.laplace_smooth(4, 0.5, Weighting::Uniform, false, false);
    assert!(spread(&mesh) < before);
}

fn square_about(center: Coordinate<3>) -> Mesh<3> {
    Mesh::from((
        vec![Connectivity::Triangular(
            vec![[0_usize, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]].into(),
        )],
        Coordinates::from([
            Coordinate::const_from([0.0, 0.0, 0.0]),
            Coordinate::const_from([2.0, 0.0, 0.0]),
            Coordinate::const_from([2.0, 2.0, 0.0]),
            Coordinate::const_from([0.0, 2.0, 0.0]),
            center,
        ]),
    ))
}

#[test]
fn preserve_boundary_ignores_interior_neighbors() -> Result<(), AssertionError> {
    let mut preserved_a = square_about([1.0, 1.0, 0.0].into());
    let mut preserved_b = square_about([1.5, 0.5, 0.0].into());
    preserved_a.laplace_smooth(1, 1.0, Weighting::Uniform, true, false);
    preserved_b.laplace_smooth(1, 1.0, Weighting::Uniform, true, false);
    assert_eq_within_tols(&preserved_a.coordinates()[0], &[1.0, 1.0, 0.0].into())?;
    assert_eq_within_tols(&preserved_a.coordinates()[0], &preserved_b.coordinates()[0])?;
    let mut free = square_about([1.5, 0.5, 0.0].into());
    free.laplace_smooth(1, 1.0, Weighting::Uniform, false, false);
    assert!(
        (free.coordinates()[0][0] - preserved_b.coordinates()[0][0]).abs() > 1e-6
            || (free.coordinates()[0][1] - preserved_b.coordinates()[0][1]).abs() > 1e-6
    );
    Ok(())
}

fn two_block_strip(corner: Coordinate<3>) -> Mesh<3> {
    Mesh::from((
        vec![
            Connectivity::Triangular(vec![[0_usize, 1, 4], [0, 4, 3]].into()),
            Connectivity::Triangular(vec![[1_usize, 2, 5], [1, 5, 4]].into()),
        ],
        Coordinates::from([
            corner,
            Coordinate::const_from([1.0, 0.0, 0.0]),
            Coordinate::const_from([2.0, 0.0, 0.0]),
            Coordinate::const_from([0.0, 1.0, 0.0]),
            Coordinate::const_from([1.0, 1.0, 0.0]),
            Coordinate::const_from([2.0, 1.0, 0.0]),
        ]),
    ))
}

#[test]
fn preserve_interfaces_ignores_off_interface_neighbors() -> Result<(), AssertionError> {
    let mut interface_a = two_block_strip([0.0, 0.0, 0.0].into());
    let mut interface_b = two_block_strip([-1.0, -1.0, 0.0].into());
    interface_a.laplace_smooth(1, 1.0, Weighting::Uniform, false, true);
    interface_b.laplace_smooth(1, 1.0, Weighting::Uniform, false, true);
    assert_eq_within_tols(&interface_a.coordinates()[1], &[1.0, 1.0, 0.0].into())?;
    assert_eq_within_tols(&interface_a.coordinates()[1], &interface_b.coordinates()[1])?;
    let mut free = two_block_strip([-1.0, -1.0, 0.0].into());
    free.laplace_smooth(1, 1.0, Weighting::Uniform, false, false);
    assert!(
        (free.coordinates()[1][0] - interface_b.coordinates()[1][0]).abs() > 1e-6
            || (free.coordinates()[1][1] - interface_b.coordinates()[1][1]).abs() > 1e-6
    );
    Ok(())
}

#[test]
fn cotangent_full_step() -> Result<(), AssertionError> {
    let mut mesh = tri();
    mesh.laplace_smooth(1, 1.0, Weighting::Cotangent, false, false);
    let coordinates = mesh.coordinates();
    assert_eq_within_tols(&coordinates[0], &[1.0, 1.0, 0.0].into())?;
    assert_eq_within_tols(&coordinates[1], &[0.0, 0.0, 0.0].into())?;
    assert_eq_within_tols(&coordinates[2], &[0.0, 0.0, 0.0].into())
}
