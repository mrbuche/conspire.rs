use super::{EqualityConstraint, Prescription};
use crate::{
    math::{Current, Matrix, Quantity, Solution, TensorRank1, TensorTupleVec, Vector},
    units::{Length, Temperature},
};

fn indices(constraint: EqualityConstraint) -> Vec<usize> {
    match constraint {
        EqualityConstraint::Fixed(indices) => indices,
        _ => panic!("expected a fixed constraint"),
    }
}

fn vector() -> Vector {
    let mut vector = Vector::zero(4);
    (0..4).for_each(|i| vector.set_flat_entry(i, i as f64 + 1.0));
    vector
}

fn entries(vector: &Vector) -> Vec<f64> {
    (0..4).map(|i| vector.flat_entry(i)).collect()
}

#[test]
fn an_offset_moves_the_initial_guess_by_an_amount() {
    let mut guess = vector();
    let constraint = EqualityConstraint::Prescribed(vec![
        (1, Prescription::Offset(0.5)),
        (3, Prescription::Offset(-1.0)),
    ])
    .lower(&mut guess);
    assert_eq!(indices(constraint), [1, 3]);
    assert_eq!(entries(&guess), [1.0, 2.5, 3.0, 3.0]);
}

#[test]
fn an_absolute_value_moves_the_initial_guess_to_it() {
    let mut guess = vector();
    let constraint =
        EqualityConstraint::Prescribed(vec![(0, Prescription::Absolute(9.0))]).lower(&mut guess);
    assert_eq!(indices(constraint), [0]);
    assert_eq!(entries(&guess), [9.0, 2.0, 3.0, 4.0]);
}

#[test]
fn an_initial_value_is_held_where_it_is() {
    let mut guess = vector();
    let constraint =
        EqualityConstraint::Prescribed(vec![(2, Prescription::Initial)]).lower(&mut guess);
    assert_eq!(indices(constraint), [2]);
    assert_eq!(entries(&guess), [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn the_last_entry_for_an_index_wins() {
    let mut guess = vector();
    let constraint = EqualityConstraint::Prescribed(vec![
        (2, Prescription::Absolute(9.0)),
        (2, Prescription::Offset(1.0)),
    ])
    .lower(&mut guess);
    assert_eq!(indices(constraint), [2]);
    assert_eq!(entries(&guess), [1.0, 2.0, 4.0, 4.0]);
}

#[test]
fn entries_for_an_index_do_not_add_up() {
    let mut guess = vector();
    EqualityConstraint::Prescribed(vec![
        (1, Prescription::Offset(1.0)),
        (1, Prescription::Offset(1.0)),
    ])
    .lower(&mut guess);
    assert_eq!(entries(&guess), [1.0, 3.0, 3.0, 4.0]);
}

#[test]
fn the_indices_are_in_ascending_order() {
    let mut guess = vector();
    let constraint = EqualityConstraint::Prescribed(vec![
        (3, Prescription::Initial),
        (0, Prescription::Initial),
        (2, Prescription::Initial),
    ])
    .lower(&mut guess);
    assert_eq!(indices(constraint), [0, 2, 3]);
}

#[test]
fn other_constraints_are_left_as_they_are() {
    let mut guess = vector();
    match EqualityConstraint::Fixed(vec![2, 0]).lower(&mut guess) {
        EqualityConstraint::Fixed(indices) => assert_eq!(indices, [2, 0]),
        _ => panic!("expected a fixed constraint"),
    }
    assert!(matches!(
        EqualityConstraint::None.lower(&mut guess),
        EqualityConstraint::None
    ));
    assert!(matches!(
        EqualityConstraint::Linear(Matrix::zero(1, 4), Vector::zero(1)).lower(&mut guess),
        EqualityConstraint::Linear(..)
    ));
    assert_eq!(entries(&guess), [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn coordinates_and_temperatures_are_moved_by_their_index() {
    let mut guess = TensorTupleVec::<TensorRank1<3, Current, Length>, Quantity<Temperature>>::from(
        Vector::zero(8),
    );
    let constraint = EqualityConstraint::Prescribed(vec![
        (7, Prescription::Absolute(300.0)),
        (5, Prescription::Offset(0.25)),
    ])
    .lower(&mut guess);
    assert_eq!(indices(constraint), [5, 7]);
    assert_eq!(guess[1].1.value(), 300.0);
    assert_eq!(guess[1].0[1].value(), 0.25);
    assert_eq!(guess[0].1.value(), 0.0);
}

#[test]
fn a_fixed_constraint_holds_its_indices_as_they_are() {
    assert_eq!(
        EqualityConstraint::Fixed(vec![2, 0, 2]).held_indices(),
        Some(vec![2, 0, 2])
    );
}

#[test]
fn a_prescribed_constraint_holds_its_indices_once_in_ascending_order() {
    assert_eq!(
        EqualityConstraint::Prescribed(vec![
            (3, Prescription::Initial),
            (1, Prescription::Absolute(2.0)),
            (3, Prescription::Offset(1.0)),
        ])
        .held_indices(),
        Some(vec![1, 3])
    );
}

#[test]
fn other_constraints_hold_no_indices() {
    assert_eq!(EqualityConstraint::None.held_indices(), None);
    assert_eq!(
        EqualityConstraint::Linear(Matrix::zero(1, 4), Vector::zero(1)).held_indices(),
        None
    );
}
