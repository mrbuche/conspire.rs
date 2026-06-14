use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh, differential::laplace::Weighting},
    },
    math::test::{TestError, assert_eq_within_tols},
};

fn triangle(coordinates: [Coordinate<3>; 3]) -> Mesh<3> {
    let connectivities = vec![Connectivity::Triangular(vec![[0_usize, 1, 2]].into())];
    Mesh::from((connectivities, Coordinates::from(coordinates)))
}

#[test]
fn single_triangle() -> Result<(), TestError> {
    let laplacian = triangle([
        Coordinate::const_from([0.0, 0.0, 0.0]),
        Coordinate::const_from([2.0, 0.0, 0.0]),
        Coordinate::const_from([0.0, 2.0, 0.0]),
    ])
    .laplacian(Weighting::Uniform);
    assert_eq_within_tols(&laplacian[0], &[-1.0, -1.0, 0.0].into())?;
    assert_eq_within_tols(&laplacian[1], &[2.0, -1.0, 0.0].into())?;
    assert_eq_within_tols(&laplacian[2], &[-1.0, 2.0, 0.0].into())
}

#[test]
fn vertex_at_neighbor_centroid_is_fixed() -> Result<(), TestError> {
    let laplacian = triangle([
        Coordinate::const_from([0.0, 0.0, 0.0]),
        Coordinate::const_from([2.0, 0.0, 0.0]),
        Coordinate::const_from([1.0, 0.0, 0.0]),
    ])
    .laplacian(Weighting::Uniform);
    assert_eq_within_tols(&laplacian[2], &[0.0, 0.0, 0.0].into())
}

#[test]
fn translation_invariant() -> Result<(), TestError> {
    let base = triangle([
        Coordinate::const_from([0.0, 0.0, 0.0]),
        Coordinate::const_from([2.0, 0.0, 0.0]),
        Coordinate::const_from([0.0, 2.0, 0.0]),
    ])
    .laplacian(Weighting::Uniform);
    let shifted = triangle([
        Coordinate::const_from([5.0, -3.0, 7.0]),
        Coordinate::const_from([7.0, -3.0, 7.0]),
        Coordinate::const_from([5.0, -1.0, 7.0]),
    ])
    .laplacian(Weighting::Uniform);
    (0..3).try_for_each(|n| assert_eq_within_tols(&base[n], &shifted[n]))
}

#[test]
fn cotangent_single_right_triangle() -> Result<(), TestError> {
    // Right angle at vertex 0; opposite cot = 0, the two 45 deg cots = 1.
    let laplacian = triangle([
        Coordinate::const_from([0.0, 0.0, 0.0]),
        Coordinate::const_from([2.0, 0.0, 0.0]),
        Coordinate::const_from([0.0, 2.0, 0.0]),
    ])
    .laplacian(Weighting::Cotangent);
    assert_eq_within_tols(&laplacian[0], &[-1.0, -1.0, 0.0].into())?;
    assert_eq_within_tols(&laplacian[1], &[2.0, 0.0, 0.0].into())?;
    assert_eq_within_tols(&laplacian[2], &[0.0, 2.0, 0.0].into())
}

#[test]
fn cotangent_translation_invariant() -> Result<(), TestError> {
    let base = triangle([
        Coordinate::const_from([0.0, 0.0, 0.0]),
        Coordinate::const_from([2.0, 0.0, 0.0]),
        Coordinate::const_from([0.0, 2.0, 0.0]),
    ])
    .laplacian(Weighting::Cotangent);
    let shifted = triangle([
        Coordinate::const_from([5.0, -3.0, 7.0]),
        Coordinate::const_from([7.0, -3.0, 7.0]),
        Coordinate::const_from([5.0, -1.0, 7.0]),
    ])
    .laplacian(Weighting::Cotangent);
    (0..3).try_for_each(|n| assert_eq_within_tols(&base[n], &shifted[n]))
}
