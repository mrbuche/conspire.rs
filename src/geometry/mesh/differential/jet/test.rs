use super::fit_jet;
use crate::{
    geometry::Coordinate,
    math::test::{TestError, assert_eq_within_tols},
};

#[test]
fn paraboloid_recovers_exact_curvatures() -> Result<(), TestError> {
    let center = [0.0, 0.0, 0.0].into();
    let neighbors = [
        [1.0, 0.0, 0.2],
        [-1.0, 0.0, 0.2],
        [0.0, 1.0, 0.05],
        [0.0, -1.0, 0.05],
        [1.0, 1.0, 0.25],
        [-1.0, 1.0, 0.25],
        [1.0, -1.0, 0.25],
        [-1.0, -1.0, 0.25],
    ]
    .map(Coordinate::from);
    let jet = fit_jet(&center, &neighbors, &[0.0, 0.0, 1.0].into()).unwrap();
    assert!((jet.principal_curvatures[0] - 0.4).abs() < 1.0e-10);
    assert!((jet.principal_curvatures[1] - 0.1).abs() < 1.0e-10);
    assert_eq_within_tols(&jet.normal, &[0.0, 0.0, 1.0].into())
}

#[test]
fn sphere_has_uniform_curvature() {
    let r = 2.0;
    let center = [0.0, 0.0, r].into();
    let mut neighbors = Vec::new();
    for theta in [0.15_f64, 0.3] {
        for k in 0..5 {
            let phi = k as f64 * std::f64::consts::TAU / 5.0;
            neighbors.push(
                [
                    r * theta.sin() * phi.cos(),
                    r * theta.sin() * phi.sin(),
                    r * theta.cos(),
                ]
                .into(),
            );
        }
    }
    let jet = fit_jet(&center, &neighbors, &[0.0, 0.0, 1.0].into()).unwrap();
    assert!((jet.max_abs_curvature() - 1.0 / r).abs() < 0.05);
    assert!(
        jet.principal_curvatures
            .iter()
            .all(|&curvature| curvature < 0.0)
    );
}
