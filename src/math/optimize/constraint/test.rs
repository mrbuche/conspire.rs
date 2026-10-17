use super::{EqualityConstraint, Prescription, Rows};
use crate::{
    math::{Current, Matrix, Quantity, Solution, Tensor, TensorRank1, TensorTupleVec, Vector},
    units::{Length, Temperature},
};

fn indices(mut constraints: Vec<EqualityConstraint>) -> Vec<usize> {
    assert_eq!(constraints.len(), 1);
    match constraints.pop() {
        Some(EqualityConstraint::Fixed(indices)) => indices,
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

fn rows(row: [f64; 4], rhs: f64) -> EqualityConstraint {
    let mut matrix = Matrix::zero(1, 4);
    (0..4).for_each(|j| matrix[0][j] = row[j]);
    EqualityConstraint::Linear(matrix, Vector::from([rhs]))
}

#[test]
fn an_offset_moves_the_initial_guess_by_an_amount() {
    let mut guess = vector();
    let constraints = EqualityConstraint::lower(
        vec![EqualityConstraint::Prescribed(vec![
            (1, Prescription::Offset(0.5)),
            (3, Prescription::Offset(-1.0)),
        ])],
        &mut guess,
    );
    assert_eq!(indices(constraints), [1, 3]);
    assert_eq!(entries(&guess), [1.0, 2.5, 3.0, 3.0]);
}

#[test]
fn an_absolute_value_moves_the_initial_guess_to_it() {
    let mut guess = vector();
    let constraints = EqualityConstraint::lower(
        vec![EqualityConstraint::Prescribed(vec![(
            0,
            Prescription::Absolute(9.0),
        )])],
        &mut guess,
    );
    assert_eq!(indices(constraints), [0]);
    assert_eq!(entries(&guess), [9.0, 2.0, 3.0, 4.0]);
}

#[test]
fn an_initial_value_is_held_where_it_is() {
    let mut guess = vector();
    let constraints = EqualityConstraint::lower(
        vec![EqualityConstraint::Prescribed(vec![(
            2,
            Prescription::Initial,
        )])],
        &mut guess,
    );
    assert_eq!(indices(constraints), [2]);
    assert_eq!(entries(&guess), [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn the_last_entry_for_an_index_wins() {
    let mut guess = vector();
    let constraints = EqualityConstraint::lower(
        vec![EqualityConstraint::Prescribed(vec![
            (2, Prescription::Absolute(9.0)),
            (2, Prescription::Offset(1.0)),
        ])],
        &mut guess,
    );
    assert_eq!(indices(constraints), [2]);
    assert_eq!(entries(&guess), [1.0, 2.0, 4.0, 4.0]);
}

#[test]
fn entries_for_an_index_do_not_add_up() {
    let mut guess = vector();
    EqualityConstraint::lower(
        vec![EqualityConstraint::Prescribed(vec![
            (1, Prescription::Offset(1.0)),
            (1, Prescription::Offset(1.0)),
        ])],
        &mut guess,
    );
    assert_eq!(entries(&guess), [1.0, 3.0, 3.0, 4.0]);
}

#[test]
fn the_indices_are_in_ascending_order() {
    let mut guess = vector();
    let constraints = EqualityConstraint::lower(
        vec![EqualityConstraint::Prescribed(vec![
            (3, Prescription::Initial),
            (0, Prescription::Initial),
            (2, Prescription::Initial),
        ])],
        &mut guess,
    );
    assert_eq!(indices(constraints), [0, 2, 3]);
}

#[test]
fn a_single_constraint_that_is_not_prescribed_is_left_as_it_is() {
    let mut guess = vector();
    assert_eq!(
        indices(EqualityConstraint::lower(
            vec![EqualityConstraint::Fixed(vec![2, 0])],
            &mut guess
        )),
        [2, 0]
    );
    assert!(matches!(
        EqualityConstraint::lower(vec![EqualityConstraint::None], &mut guess).as_slice(),
        [EqualityConstraint::None]
    ));
    assert!(matches!(
        EqualityConstraint::lower(vec![rows([1.0; 4], 0.0)], &mut guess).as_slice(),
        [EqualityConstraint::Linear(..)]
    ));
    assert_eq!(entries(&guess), [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn coordinates_and_temperatures_are_moved_by_their_index() {
    let mut guess = TensorTupleVec::<TensorRank1<3, Current, Length>, Quantity<Temperature>>::from(
        Vector::zero(8),
    );
    let constraints = EqualityConstraint::lower(
        vec![EqualityConstraint::Prescribed(vec![
            (7, Prescription::Absolute(300.0)),
            (5, Prescription::Offset(0.25)),
        ])],
        &mut guess,
    );
    assert_eq!(indices(constraints), [5, 7]);
    assert_eq!(guess[1].1.value(), 300.0);
    assert_eq!(guess[1].0[1].value(), 0.25);
    assert_eq!(guess[0].1.value(), 0.0);
}

#[test]
fn a_fixed_constraint_holds_its_indices_as_they_are() {
    assert_eq!(
        EqualityConstraint::held_indices(&[EqualityConstraint::Fixed(vec![2, 0, 2])]),
        Some(vec![2, 0, 2])
    );
}

#[test]
fn a_prescribed_constraint_holds_its_indices_once_in_ascending_order() {
    assert_eq!(
        EqualityConstraint::held_indices(&[EqualityConstraint::Prescribed(vec![
            (3, Prescription::Initial),
            (1, Prescription::Absolute(2.0)),
            (3, Prescription::Offset(1.0)),
        ])]),
        Some(vec![1, 3])
    );
}

#[test]
fn other_constraints_hold_no_indices() {
    assert_eq!(
        EqualityConstraint::held_indices(&[EqualityConstraint::None]),
        None
    );
    assert_eq!(EqualityConstraint::held_indices(&[]), None);
    assert_eq!(
        EqualityConstraint::held_indices(&[rows([1.0; 4], 0.0)]),
        None
    );
    assert_eq!(
        EqualityConstraint::held_indices(&[EqualityConstraint::Multipliers(vec![(
            0,
            Prescription::Initial
        )])]),
        None
    );
}

mod several {
    use super::*;
    #[test]
    fn fixed_and_prescribed_are_merged_with_the_last_entry_winning() {
        let mut guess = vector();
        let constraints = EqualityConstraint::lower(
            vec![
                EqualityConstraint::Fixed(vec![2, 0]),
                EqualityConstraint::Prescribed(vec![(0, Prescription::Absolute(7.0))]),
                EqualityConstraint::Prescribed(vec![(3, Prescription::Offset(1.0))]),
            ],
            &mut guess,
        );
        assert_eq!(indices(constraints), [0, 2, 3]);
        assert_eq!(entries(&guess), [7.0, 2.0, 3.0, 5.0]);
    }
    #[test]
    fn a_later_hold_undoes_an_earlier_value() {
        let mut guess = vector();
        let constraints = EqualityConstraint::lower(
            vec![
                EqualityConstraint::Prescribed(vec![(1, Prescription::Absolute(9.0))]),
                EqualityConstraint::Fixed(vec![1]),
            ],
            &mut guess,
        );
        assert_eq!(indices(constraints), [1]);
        assert_eq!(entries(&guess), [1.0, 2.0, 3.0, 4.0]);
    }
    #[test]
    fn linear_constraints_are_stacked() {
        let merged = EqualityConstraint::merged(vec![
            rows([1.0, 0.0, 0.0, 0.0], 5.0),
            EqualityConstraint::None,
            rows([0.0, 2.0, 0.0, 3.0], 6.0),
        ]);
        let [EqualityConstraint::Linear(matrix, rhs)] = merged.as_slice() else {
            panic!("expected a linear constraint")
        };
        assert_eq!((matrix.len(), matrix.width()), (2, 4));
        assert_eq!((matrix[0][0], matrix[1][1], matrix[1][3]), (1.0, 2.0, 3.0));
        assert_eq!((rhs[0], rhs[1]), (5.0, 6.0));
    }
    #[test]
    fn nothing_merges_to_nothing_and_a_single_constraint_to_itself() {
        assert!(EqualityConstraint::merged(vec![]).is_empty());
        assert!(
            EqualityConstraint::merged(vec![EqualityConstraint::None, EqualityConstraint::None])
                .is_empty()
        );
        assert!(matches!(
            EqualityConstraint::merged(vec![rows([1.0; 4], 0.0)]).as_slice(),
            [EqualityConstraint::Linear(..)]
        ));
        assert!(matches!(
            EqualityConstraint::merged(vec![EqualityConstraint::None]).as_slice(),
            [EqualityConstraint::None]
        ));
    }
    #[test]
    fn prescribed_with_linear_is_lowered_to_fixed_with_linear() {
        let mut guess = vector();
        let parts = EqualityConstraint::lower(
            vec![
                EqualityConstraint::Prescribed(vec![(1, Prescription::Absolute(9.0))]),
                rows([1.0; 4], 0.0),
            ],
            &mut guess,
        );
        assert_eq!(parts.len(), 2);
        assert!(matches!(&parts[0], EqualityConstraint::Fixed(i) if i == &[1]));
        assert!(matches!(&parts[1], EqualityConstraint::Linear(..)));
        assert_eq!(entries(&guess), [1.0, 9.0, 3.0, 4.0]);
    }
    #[test]
    fn held_indices_see_through_several_unless_linear_is_among_them() {
        assert_eq!(
            EqualityConstraint::held_indices(&[
                EqualityConstraint::Fixed(vec![3]),
                EqualityConstraint::Prescribed(vec![(1, Prescription::Initial)]),
            ]),
            Some(vec![1, 3])
        );
        assert_eq!(
            EqualityConstraint::held_indices(&[
                EqualityConstraint::Fixed(vec![3]),
                rows([1.0; 4], 0.0),
            ]),
            None
        );
    }
    #[test]
    fn the_last_way_an_index_is_held_wins() {
        let merged = EqualityConstraint::merged(vec![
            EqualityConstraint::Prescribed(vec![(0, Prescription::Absolute(1.0))]),
            EqualityConstraint::Multipliers(vec![(0, Prescription::Absolute(2.0))]),
        ]);
        let [EqualityConstraint::Multipliers(entries)] = merged.as_slice() else {
            panic!("expected a constraint held by multipliers")
        };
        assert_eq!(entries, &[(0, Prescription::Absolute(2.0))]);
        let merged = EqualityConstraint::merged(vec![
            EqualityConstraint::Multipliers(vec![(0, Prescription::Absolute(2.0))]),
            EqualityConstraint::Fixed(vec![0]),
        ]);
        let [EqualityConstraint::Prescribed(entries)] = merged.as_slice() else {
            panic!("expected a prescribed constraint")
        };
        assert_eq!(entries, &[(0, Prescription::Initial)]);
    }
    #[test]
    fn splitting_eliminates_some_and_makes_rows_of_the_rest_in_ascending_order() {
        let mut guess = vector();
        let (eliminated, Rows::Sparse(matrix, rhs)) = EqualityConstraint::split(
            vec![
                EqualityConstraint::Prescribed(vec![(0, Prescription::Absolute(7.0))]),
                EqualityConstraint::Multipliers(vec![
                    (3, Prescription::Offset(0.5)),
                    (1, Prescription::Initial),
                    (2, Prescription::Absolute(9.0)),
                ]),
                rows([0.0, 1.0, 0.0, 2.0], 6.0),
            ],
            &mut guess,
        ) else {
            panic!("expected sparse rows")
        };
        assert_eq!(eliminated, Some(vec![0]));
        assert_eq!(entries(&guess), [7.0, 2.0, 3.0, 4.0]);
        assert_eq!(rhs.len(), 4);
        assert_eq!((rhs[0], rhs[1], rhs[2], rhs[3]), (2.0, 9.0, 4.5, 6.0));
        let row_of = |row| (0..4).map(|j| matrix.entry(row, j)).collect::<Vec<_>>();
        assert_eq!(row_of(0), [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(row_of(1), [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(row_of(2), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(row_of(3), [0.0, 1.0, 0.0, 2.0]);
    }
    #[test]
    fn splitting_without_multiplier_entries_leaves_a_linear_constraint_dense() {
        let mut guess = vector();
        let (eliminated, split) = EqualityConstraint::split(vec![rows([1.0; 4], 3.0)], &mut guess);
        assert_eq!(eliminated, None);
        assert!(matches!(split, Rows::Dense(..)));
        let (eliminated, split) =
            EqualityConstraint::split(vec![EqualityConstraint::None], &mut guess);
        assert_eq!(eliminated, None);
        assert!(matches!(split, Rows::None));
        let (eliminated, split) = EqualityConstraint::split(vec![], &mut guess);
        assert_eq!(eliminated, None);
        assert!(matches!(split, Rows::None));
        let (eliminated, split) =
            EqualityConstraint::split(vec![EqualityConstraint::Fixed(vec![])], &mut guess);
        assert_eq!(eliminated, Some(vec![]));
        assert!(matches!(split, Rows::None));
    }
    #[test]
    fn a_single_lowered_constraint_is_given_and_several_are_not() {
        let mut guess = vector();
        assert!(matches!(
            EqualityConstraint::lower_single(vec![], &mut guess),
            Some(EqualityConstraint::None)
        ));
        assert!(matches!(
            EqualityConstraint::lower_single(
                vec![EqualityConstraint::Prescribed(vec![(
                    1,
                    Prescription::Initial
                )])],
                &mut guess
            ),
            Some(EqualityConstraint::Fixed(_))
        ));
        assert!(
            EqualityConstraint::lower_single(
                vec![EqualityConstraint::Fixed(vec![0]), rows([1.0; 4], 0.0)],
                &mut guess
            )
            .is_none()
        );
    }
}
