use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Connectivity, Fitting, Mesh, Verdict,
            tessellation::{D, Tessellation},
        },
        ntree::{Balancing, CurvatureSizing},
    },
    math::Scalar,
};
use std::f64::consts::TAU;

fn torus(major: Scalar, minor: Scalar, around: usize, tube: usize) -> Tessellation {
    let mut coordinates = Vec::new();
    (0..around).for_each(|i| {
        let theta = TAU * i as Scalar / around as Scalar;
        (0..tube).for_each(|j| {
            let phi = TAU * j as Scalar / tube as Scalar;
            let radius = major + minor * phi.cos();
            coordinates.push([
                radius * theta.cos(),
                radius * theta.sin(),
                minor * phi.sin(),
            ])
        })
    });
    let index = |i: usize, j: usize| (i % around) * tube + (j % tube);
    let faces: Vec<[usize; D]> = (0..around)
        .flat_map(|i| {
            (0..tube).flat_map(move |j| {
                [
                    [index(i, j), index(i + 1, j), index(i + 1, j + 1)],
                    [index(i, j), index(i + 1, j + 1), index(i, j + 1)],
                ]
            })
        })
        .collect();
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(coordinates),
    )))
}

/// A slender torus at a coarse scale is where trimming back too far shows up:
/// the buffer layer ends up spanning a gap the core should have covered. The
/// `s_max/2` margin rule inverted 41 elements here (worst scaled Jacobian
/// -0.5748); the signed-distance ratio rule leaves none.
#[test]
fn dualize_slender_torus_is_not_inverted() {
    let mesh = torus(1.0, 0.15, 64, 24)
        .dualize(
            Balancing::Strong(1),
            3.0,
            CurvatureSizing::default(),
            Fitting::Snap,
        )
        .unwrap();
    let worst = mesh
        .minimum_scaled_jacobians()
        .into_iter()
        .flatten()
        .fold(Scalar::INFINITY, Scalar::min);
    assert!(worst > 0.15, "{worst}");
}
