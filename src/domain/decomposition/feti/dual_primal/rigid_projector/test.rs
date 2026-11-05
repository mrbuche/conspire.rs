use super::RigidProjector;
use crate::math::{Tensor, Vector};

fn vector(entries: &[f64]) -> Vector {
    entries.iter().copied().collect()
}

fn projector() -> RigidProjector {
    RigidProjector::from_columns(vec![
        vector(&[1.0, 0.0, 1.0, 0.0]),
        vector(&[0.0, 1.0, 1.0, 1.0]),
    ])
    .unwrap()
}

fn transposed(projector: &RigidProjector, v: &Vector) -> Vector {
    projector.restrict(v)
}

#[test]
fn a_projected_vector_has_nothing_along_the_columns() {
    let projector = projector();
    let projected = projector.project(&vector(&[3.0, -1.0, 2.0, 5.0]));
    transposed(&projector, &projected)
        .iter()
        .for_each(|&entry| assert!(entry.abs() < 1e-12));
}

#[test]
fn projecting_twice_is_projecting_once() {
    let projector = projector();
    let once = projector.project(&vector(&[3.0, -1.0, 2.0, 5.0]));
    let twice = projector.project(&once);
    (0..4).for_each(|row| assert!((once[row] - twice[row]).abs() < 1e-12));
}

#[test]
fn the_particular_multipliers_satisfy_the_constraint() {
    let projector = projector();
    let e = vector(&[2.0, -3.0]);
    let lambda = projector.particular(&e, 4);
    let restricted = transposed(&projector, &lambda);
    (0..2).for_each(|row| assert!((restricted[row] - e[row]).abs() < 1e-12));
}

#[test]
fn amplitudes_recover_a_combination_of_the_columns() {
    let projector = projector();
    let combination = projector.particular(&vector(&[1.0, 0.0]), 4);
    let amplitudes = projector.amplitudes(&combination);
    let expected = projector.expand(&amplitudes, 4);
    (0..4).for_each(|row| assert!((combination[row] - expected[row]).abs() < 1e-12));
}

#[test]
fn dependent_columns_are_refused() {
    assert!(
        RigidProjector::from_columns(vec![vector(&[1.0, 2.0]), vector(&[-1.0, -2.0])]).is_none()
    );
}

#[test]
fn no_columns_project_nothing() {
    let projector = RigidProjector::from_columns(Vec::new()).unwrap();
    assert_eq!(projector.len(), 0);
    let v = vector(&[1.0, 2.0]);
    assert_eq!(projector.project(&v)[1], 2.0);
}
