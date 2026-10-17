use crate::{
    domain::boundary::{BoundaryConditions, ConstrainAt, SolidRate, Velocity},
    geometry::mesh::NodeSet,
    math::{
        Quantity,
        optimize::{EqualityConstraint, Prescription},
    },
    units::{Time, Velocity as VelocityUnit},
};

fn at(conditions: &impl ConstrainAt<SolidRate<3>>, time: f64) -> Vec<(usize, Prescription)> {
    let mut constraints = conditions.constraint_at(Quantity::<Time>::new(time));
    match constraints.pop() {
        Some(EqualityConstraint::Prescribed(held)) => held,
        _ => panic!("expected a prescribed constraint last"),
    }
}

#[test]
fn a_value_follows_time() {
    let conditions = BoundaryConditions::<SolidRate<3>>::new().prescribe_nodes_with(
        Velocity::<1>,
        [0, 2],
        |time| VelocityUnit::meters_per_second(2.0 * time.value()),
    );
    assert_eq!(
        at(&conditions, 1.5),
        [
            (1, Prescription::Absolute(3.0)),
            (7, Prescription::Absolute(3.0))
        ]
    );
    assert_eq!(
        at(&conditions, 4.0),
        [
            (1, Prescription::Absolute(8.0)),
            (7, Prescription::Absolute(8.0))
        ]
    );
}

#[test]
fn the_static_conditions_come_first() {
    let set = NodeSet::new(vec![1]);
    let conditions = BoundaryConditions::<SolidRate<3>>::new()
        .fix(Velocity::<2>, &set)
        .prescribe_nodes_with(Velocity::<0>, [1], |time| {
            VelocityUnit::meters_per_second(time.value())
        });
    let constraints = conditions.constraint_at(Quantity::<Time>::new(0.5));
    assert_eq!(constraints.len(), 2);
    assert!(matches!(&constraints[0], EqualityConstraint::Fixed(held) if held == &[5]));
    assert!(matches!(
        &constraints[1],
        EqualityConstraint::Prescribed(held) if held == &[(3, Prescription::Absolute(0.5))]
    ));
}

#[test]
fn a_set_may_be_held() {
    let set = NodeSet::new(vec![2]);
    let conditions =
        BoundaryConditions::<SolidRate<3>>::new().prescribe_with(Velocity::<0>, &set, |time| {
            VelocityUnit::meters_per_second(-time.value())
        });
    assert_eq!(at(&conditions, 2.0), [(6, Prescription::Absolute(-2.0))]);
}
