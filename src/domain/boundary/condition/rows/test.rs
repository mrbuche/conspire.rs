use crate::{
    domain::boundary::{Constrain, ConstrainAt, Displacement, Enforcement, Rows, Solid},
    math::{
        Quantity,
        optimize::{EqualityConstraint, Prescription},
    },
    units::Length,
};

#[test]
fn a_runtime_number_of_nodes_may_be_held() {
    let rows = (0..4).fold(
        Rows::<Solid<3>>::new(Enforcement::Multipliers),
        |rows, node| {
            rows.add(
                Displacement::<2>,
                node,
                Quantity::<Length>::new(node as f64),
            )
        },
    );
    let constraints = rows.constraint();
    assert_eq!(constraints.len(), 1);
    let EqualityConstraint::Multipliers(held) = &constraints[0] else {
        panic!("expected multipliers")
    };
    assert_eq!(
        held,
        &[
            (2, Prescription::Offset(0.0)),
            (5, Prescription::Offset(1.0)),
            (8, Prescription::Offset(2.0)),
            (11, Prescription::Offset(3.0)),
        ]
    );
}

#[test]
fn rows_eliminated_at_their_initial_values_are_fixed() {
    let rows = Rows::<Solid<3>>::new(Enforcement::Elimination)
        .hold(7, Prescription::Initial)
        .hold(1, Prescription::Initial)
        .hold(7, Prescription::Initial);
    assert!(matches!(
        &rows.constraint()[0],
        EqualityConstraint::Fixed(held) if held == &[1, 7]
    ));
}

#[test]
fn rows_may_be_followed_by_other_conditions() {
    let rows = Rows::<Solid<3>>::new(Enforcement::Multipliers).hold(0, Prescription::Initial);
    let conditions = rows.fix_node(Displacement::<1>, 2);
    assert_eq!(conditions.constraint().len(), 2);
}
