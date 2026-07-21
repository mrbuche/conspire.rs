use super::{energy, scatter};
use crate::{
    EPSILON,
    geometry::{Coordinate, Coordinates},
    math::assert::{Assert, AssertionError},
};
use std::array::from_fn;

#[test]
fn gradient_matches_finite_difference() -> Result<(), AssertionError> {
    let hex = from_fn(|i| i);
    let mut coordinates = Coordinates::from(vec![
        [0.03, -0.04, 0.01],
        [1.08, 0.05, -0.07],
        [1.02, 0.94, 0.11],
        [-0.06, 1.07, 0.02],
        [0.09, 0.01, 0.88],
        [0.94, -0.08, 1.04],
        [1.11, 1.03, 0.93],
        [0.05, 0.92, 1.09],
    ]);
    for epsilon in [1.0, 1.0e-3] {
        let scattered = scatter(&hex, &coordinates, epsilon);
        for node in 0..8 {
            let analytic = scattered[node].clone();
            let numerical = Coordinate::from(from_fn(|i| {
                coordinates[node][i] += EPSILON;
                let above = energy(&hex, &coordinates, epsilon);
                coordinates[node][i] -= 2.0 * EPSILON;
                let below = energy(&hex, &coordinates, epsilon);
                coordinates[node][i] += EPSILON;
                (above - below) / (2.0 * EPSILON)
            }));
            Assert::default().eq_within_fd_tol(analytic, &numerical)?;
        }
    }
    Ok(())
}
