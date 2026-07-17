use crate::math::assert::Assert;
use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh, differential::laplace::Weighting},
    },
    math::{Scalar, Tensor, assert::AssertionError},
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

fn spread(mesh: &Mesh<3>) -> Scalar {
    let coordinates = mesh.coordinates();
    let center = mesh.coordinates().iter().sum::<Coordinate<3>>() / 3.0;
    (0..3)
        .map(|node| {
            (0..3)
                .map(|i| (coordinates[node][i] - center[i]).powi(2))
                .sum::<Scalar>()
        })
        .sum()
}

#[test]
fn zero_iterations_is_identity() -> Result<(), AssertionError> {
    let mut mesh = tri();
    mesh.taubin_smooth(0, 0.1, 0.5, Weighting::Uniform, false, false);
    let coordinates = mesh.coordinates();
    Assert::default().eq_within_tols(&coordinates[0], &[0.0, 0.0, 0.0].into())?;
    Assert::default().eq_within_tols(&coordinates[1], &[2.0, 0.0, 0.0].into())?;
    Assert::default().eq_within_tols(&coordinates[2], &[0.0, 2.0, 0.0].into())
}

#[test]
fn first_iteration_matches_laplace_deflate() -> Result<(), AssertionError> {
    let mut laplace = tri();
    laplace.laplace_smooth(1, 0.5, Weighting::Uniform, false, false);
    let mut taubin = tri();
    taubin.taubin_smooth(1, 0.1, 0.5, Weighting::Uniform, false, false);
    (0..3).try_for_each(|node| {
        Assert::default().eq_within_tols(&laplace.coordinates()[node], &taubin.coordinates()[node])
    })
}

#[test]
fn resists_shrinkage_relative_to_laplace() {
    let mut laplace = tri();
    laplace.laplace_smooth(2, 0.5, Weighting::Uniform, false, false);
    let mut taubin = tri();
    taubin.taubin_smooth(2, 0.1, 0.5, Weighting::Uniform, false, false);
    assert!(spread(&taubin) > spread(&laplace));
}
