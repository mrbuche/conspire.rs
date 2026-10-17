use super::Timewise;
use crate::{
    domain::boundary::{ConstrainAt, Displacement, Solid},
    math::{
        Quantity,
        optimize::{EqualityConstraint, Prescription},
    },
    units::Time,
};

#[test]
fn constraints_follow_the_function_of_time() {
    let conditions = Timewise(|time: Quantity<Time>| {
        vec![EqualityConstraint::Prescribed(vec![(
            2,
            Prescription::Absolute(3.0 * time.value()),
        )])]
    });
    let at = |time: f64| {
        let constraints =
            ConstrainAt::<Solid<3>>::constraint_at(&conditions, Quantity::<Time>::new(time));
        let [EqualityConstraint::Prescribed(held)] = constraints.as_slice() else {
            panic!("expected one prescribed constraint")
        };
        held.clone()
    };
    assert_eq!(at(1.0), [(2, Prescription::Absolute(3.0))]);
    assert_eq!(at(2.5), [(2, Prescription::Absolute(7.5))]);
}

#[test]
fn other_conditions_may_follow() {
    let conditions = Timewise(|_: Quantity<Time>| vec![EqualityConstraint::Fixed(vec![0])])
        .fix_node(Displacement::<1>, 1);
    let constraints =
        ConstrainAt::<Solid<3>>::constraint_at(&conditions, Quantity::<Time>::new(0.0));
    assert_eq!(constraints.len(), 2);
    assert!(matches!(&constraints[0], EqualityConstraint::Fixed(held) if held == &[0]));
    assert!(matches!(&constraints[1], EqualityConstraint::Fixed(held) if held == &[4]));
}
