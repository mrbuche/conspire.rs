use super::{eigenvalues_below, largest_eigenvalue, time_scale_exceeds};
use crate::{math::Scalar, units::Time};

fn diagonal(entries: &'static [Scalar]) -> impl Fn(usize, usize) -> Scalar {
    move |row, column| if row == column { entries[row] } else { 0.0 }
}

#[test]
fn brackets_the_eigenvalues_of_a_diagonal_pair() {
    let stiffness = diagonal(&[1.0, 4.0, 9.0]);
    let masses = [1.0; 3];
    assert!(eigenvalues_below(3, &stiffness, &masses, 9.01));
    assert!(eigenvalues_below(3, &stiffness, &masses, 100.0));
    assert!(!eigenvalues_below(3, &stiffness, &masses, 8.99));
    assert!(!eigenvalues_below(3, &stiffness, &masses, 4.0));
    assert!(!eigenvalues_below(3, &stiffness, &masses, 0.5));
}

#[test]
fn accounts_for_the_masses() {
    let stiffness = diagonal(&[1.0, 4.0, 9.0]);
    let masses = [1.0, 2.0, 9.0];
    // the eigenvalues of M⁻¹K are 1, 2, and 1
    assert!(eigenvalues_below(3, &stiffness, &masses, 2.01));
    assert!(!eigenvalues_below(3, &stiffness, &masses, 1.99));
}

#[test]
fn brackets_a_coupled_pair() {
    let stiffness = |row: usize, column: usize| if row == column { 2.0 } else { -1.0 };
    // the largest eigenvalue of M⁻¹K is (3 + √3) / 2
    let masses = [1.0, 2.0];
    let largest = (3.0 + 3.0_f64.sqrt()) / 2.0;
    assert!(eigenvalues_below(2, stiffness, &masses, largest * 1.001));
    assert!(!eigenvalues_below(2, stiffness, &masses, largest * 0.999));
}

#[test]
fn agrees_with_the_power_iteration() {
    let size = 12;
    let stiffness = |row: usize, column: usize| {
        (0..size)
            .map(|k| {
                let factor = |index: usize| ((index * 7 + k * 13 + 3) % 11) as Scalar - 5.0;
                factor(row) * factor(column)
            })
            .sum::<Scalar>()
            + if row == column {
                0.5 * (row as Scalar + 1.0)
            } else {
                0.0
            }
    };
    let masses = (0..size)
        .map(|index| 1.0 + 0.3 * index as Scalar)
        .collect::<Vec<_>>();
    let largest = largest_eigenvalue(size, stiffness, &masses);
    assert!(eigenvalues_below(size, stiffness, &masses, largest * 1.001));
    assert!(!eigenvalues_below(
        size,
        stiffness,
        &masses,
        largest * 0.999
    ));
}

#[test]
fn compares_a_time_scale() {
    let stiffness = diagonal(&[1.0, 4.0, 9.0]);
    let masses = [1.0; 3];
    // the fastest time scale is 1 / 3
    assert!(time_scale_exceeds(
        3,
        &stiffness,
        &masses,
        Time::seconds(0.33)
    ));
    assert!(!time_scale_exceeds(
        3,
        &stiffness,
        &masses,
        Time::seconds(0.34)
    ));
}
