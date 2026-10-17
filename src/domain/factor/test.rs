use super::fixed_indices;
use crate::math::{
    Matrix, Vector,
    optimize::{EqualityConstraint, Prescription},
};

#[test]
fn fixed_indices_are_the_indices() {
    assert_eq!(
        fixed_indices(vec![EqualityConstraint::Fixed(vec![2, 0])], "an integrator").unwrap(),
        [2, 0]
    );
}

#[test]
fn nothing_is_constrained_without_indices() {
    assert!(
        fixed_indices(vec![EqualityConstraint::None], "an integrator")
            .unwrap()
            .is_empty()
    );
}

#[test]
fn holds_are_indices_once_in_ascending_order() {
    let holds = EqualityConstraint::Prescribed(vec![
        (3, Prescription::Initial),
        (1, Prescription::Initial),
        (3, Prescription::Initial),
    ]);
    assert_eq!(fixed_indices(vec![holds], "an integrator").unwrap(), [1, 3]);
}

#[test]
fn values_are_not_supported() {
    let values = EqualityConstraint::Prescribed(vec![
        (1, Prescription::Initial),
        (3, Prescription::Offset(0.1)),
    ]);
    assert!(
        fixed_indices(vec![values], "an integrator")
            .unwrap_err()
            .to_string()
            .contains("an integrator")
    );
    let absolute = EqualityConstraint::Prescribed(vec![(3, Prescription::Absolute(0.1))]);
    assert!(fixed_indices(vec![absolute], "an integrator").is_err());
}

#[test]
fn multipliers_are_not_supported() {
    let linear = EqualityConstraint::Linear(Matrix::zero(1, 4), Vector::zero(1));
    assert!(fixed_indices(vec![linear], "an integrator").is_err());
}
