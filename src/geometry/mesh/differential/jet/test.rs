use super::{fit_jet, vertex_jets};
use crate::{
    geometry::{Coordinate, Coordinates},
    math::test::{TestError, assert_eq_within_tols},
};

fn flat_grid(n: usize) -> (Vec<[usize; 3]>, Coordinates<3>) {
    let coordinates = Coordinates::from(
        (0..n)
            .flat_map(|i| (0..n).map(move |j| [i as f64, j as f64, 0.0]))
            .collect::<Vec<_>>(),
    );
    let mut connectivity = Vec::new();
    for i in 0..n - 1 {
        for j in 0..n - 1 {
            let (v00, v10, v01, v11) = (
                i * n + j,
                (i + 1) * n + j,
                i * n + j + 1,
                (i + 1) * n + j + 1,
            );
            connectivity.push([v00, v10, v11]);
            connectivity.push([v00, v11, v01]);
        }
    }
    (connectivity, coordinates)
}

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

#[test]
fn flat_grid_interior_has_zero_curvature() {
    let (connectivity, coordinates) = flat_grid(5);
    let jets = vertex_jets(&connectivity, &coordinates);
    let center = jets[2 * 5 + 2]
        .as_ref()
        .expect("interior vertex should fit");
    assert!(center.max_abs_curvature() < 1.0e-9);
}
