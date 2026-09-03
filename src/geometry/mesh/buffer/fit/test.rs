use super::{energy, scatter};
use crate::math::assert::perturbation;
use crate::{
    EPSILON,
    geometry::{
        Coordinates,
        mesh::quality::metrics::{hexahedron, tetrahedron},
    },
    math::{
        Reference, TensorRank1,
        assert::{Assert, AssertionError},
    },
    units::ReciprocalLength,
};
use std::array::from_fn;

/// Checks the scattered energy gradient of one element against a central
/// difference of the energy itself.
fn gradient<const N: usize>(
    corners: &[(usize, [usize; 3]); N],
    mut coordinates: Coordinates<3>,
) -> Result<(), AssertionError> {
    let element = from_fn(|i| i);
    for epsilon in [1.0, 1.0e-3] {
        let scattered = scatter(corners, &element, &coordinates, epsilon);
        for node in 0..N {
            let analytic = scattered[node].clone();
            let numerical = TensorRank1::<3, Reference, ReciprocalLength>::from(from_fn(|i| {
                coordinates[node][i] += perturbation(EPSILON);
                let above = energy(corners, &element, &coordinates, epsilon);
                coordinates[node][i] -= perturbation(2.0 * EPSILON);
                let below = energy(corners, &element, &coordinates, epsilon);
                coordinates[node][i] += perturbation(EPSILON);
                (above - below) / (2.0 * EPSILON)
            }));
            Assert::default().eq_within_fd_tol(analytic, &numerical)?;
        }
    }
    Ok(())
}

#[test]
fn gradient_matches_finite_difference() -> Result<(), AssertionError> {
    gradient(
        &hexahedron::CORNERS,
        Coordinates::from(vec![
            [0.03, -0.04, 0.01],
            [1.08, 0.05, -0.07],
            [1.02, 0.94, 0.11],
            [-0.06, 1.07, 0.02],
            [0.09, 0.01, 0.88],
            [0.94, -0.08, 1.04],
            [1.11, 1.03, 0.93],
            [0.05, 0.92, 1.09],
        ]),
    )
}

#[test]
fn tetrahedral_gradient_matches_finite_difference() -> Result<(), AssertionError> {
    gradient(
        &tetrahedron::CORNERS,
        Coordinates::from(vec![
            [0.03, -0.04, 0.01],
            [1.08, 0.05, -0.07],
            [0.02, 0.94, 0.11],
            [0.09, 0.01, 0.88],
        ]),
    )
}
