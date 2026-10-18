use super::super::test::{indices, layout_of, sets};
use crate::{
    domain::boundary::{
        BoundaryConditions, Concentration, Constrain, ConstrainAt, Coupled, Displacement,
        Enforcement, Field, Potential, Solid, Static, Temperature,
    },
    math::{
        Quantity, Vector,
        optimize::{EqualityConstraint, EqualityConstraints, Prescription},
    },
    units::{
        Concentration as ConcentrationUnit, ElectricPotential, Length,
        Temperature as TemperatureUnit,
    },
};
use std::marker::PhantomData;

fn entries(constraints: EqualityConstraints) -> Vec<(usize, Prescription)> {
    constraints
        .into_iter()
        .flat_map(|constraint| match constraint {
            EqualityConstraint::Prescribed(entries) => entries,
            EqualityConstraint::Fixed(indices) => indices
                .into_iter()
                .map(|index| (index, Prescription::Initial))
                .collect(),
            _ => panic!("expected a fixed or prescribed constraint"),
        })
        .collect()
}

#[test]
fn a_displacement_is_held_at_a_distance_from_where_it_starts() {
    let sets = sets();
    let prescribed = BoundaryConditions::<Solid<3>>::new().prescribe(
        Displacement::<1>,
        sets.by_name("bottom").unwrap(),
        Quantity::<Length>::new(0.25),
    );
    assert_eq!(
        entries(prescribed.constraint()),
        [
            (1, Prescription::Offset(0.25)),
            (7, Prescription::Offset(0.25))
        ]
    );
}

#[test]
fn a_field_is_held_at_its_value() {
    let prescribed = BoundaryConditions::<Field<TemperatureUnit>>::new().prescribe_nodes(
        Temperature,
        [3, 1],
        Quantity::<TemperatureUnit>::new(500.0),
    );
    assert_eq!(
        entries(prescribed.constraint()),
        [
            (3, Prescription::Absolute(500.0)),
            (1, Prescription::Absolute(500.0))
        ]
    );
}

#[test]
fn every_field_is_held_at_its_value() {
    let prescribed = BoundaryConditions::<
        Coupled<Coupled<Solid<3>, Field<TemperatureUnit>>, Field<ElectricPotential>>,
    >::new()
    .prescribe_node(Temperature, 2, Quantity::<TemperatureUnit>::new(300.0))
    .prescribe_node(Potential, 2, Quantity::<ElectricPotential>::new(1.5));
    assert_eq!(
        entries(prescribed.constraint()),
        [
            (13, Prescription::Absolute(300.0)),
            (14, Prescription::Absolute(1.5))
        ]
    );
    let concentration = BoundaryConditions::<Field<ConcentrationUnit>>::new().prescribe_node(
        Concentration,
        4,
        Quantity::<ConcentrationUnit>::new(2.0),
    );
    assert_eq!(
        entries(concentration.constraint()),
        [(4, Prescription::Absolute(2.0))]
    );
}

#[test]
fn a_velocity_is_held_at_its_value_in_its_own_layout() {
    use crate::{
        domain::{
            NodalVelocities,
            boundary::{SolidRate, Velocity},
        },
        units::Velocity as VelocityUnit,
    };
    let _: PhantomData<SolidRate<3>> = layout_of::<NodalVelocities<3>>();
    let rates = BoundaryConditions::<SolidRate<3>>::new().prescribe_node(
        Velocity::<1>,
        2,
        VelocityUnit::meters_per_second(0.5),
    );
    assert_eq!(
        entries(rates.constraint()),
        [(7, Prescription::Absolute(0.5))]
    );
    let held = BoundaryConditions::<SolidRate<2>>::new().fix_node(Velocity::<1>, 3);
    assert_eq!(indices(held.constraint()), [7]);
}

#[test]
fn a_hold_among_values_is_held_at_its_initial_value() {
    let prescribed = BoundaryConditions::<Solid<3>>::new()
        .fix_node(Displacement::<0>, 0)
        .prescribe_node(Displacement::<2>, 0, Quantity::<Length>::new(0.5));
    assert_eq!(
        entries(prescribed.constraint()),
        [(0, Prescription::Initial), (2, Prescription::Offset(0.5))]
    );
}

#[test]
fn only_holds_are_fixed() {
    let held = BoundaryConditions::<Solid<3>>::new().fix_node(Displacement::<0>, 1);
    assert_eq!(indices(held.constraint()), [3]);
}

#[test]
fn the_conditions_know_what_they_hold() {
    let prescribed = BoundaryConditions::<Solid<3>>::new()
        .fix_node(Displacement::<0>, 0)
        .prescribe_node(Displacement::<1>, 0, Quantity::<Length>::new(0.5));
    assert!(matches!(
        Static::<Solid<3>>::held_constraint(&prescribed.0.1),
        EqualityConstraint::Fixed(_)
    ));
    assert_eq!(prescribed.1.prescription(), Prescription::Offset(0.5));
}

#[test]
fn a_condition_is_eliminated_unless_made_otherwise() {
    let sets = sets();
    let bcs = BoundaryConditions::<Solid<3>>::new()
        .fix_node(Displacement::<0>, 0)
        .fix_by(
            Enforcement::Multipliers,
            Displacement::<1>,
            sets.by_name("bottom").unwrap(),
        )
        .prescribe_node(Displacement::<2>, 0, Quantity::<Length>::new(0.5))
        .prescribe_nodes_by(
            Enforcement::Multipliers,
            Displacement::<2>,
            [1, 2],
            Quantity::<Length>::new(0.25),
        )
        .prescribe_node_by(
            Enforcement::Elimination,
            Displacement::<0>,
            3,
            Quantity::<Length>::new(0.125),
        );
    assert_eq!(
        [
            bcs.0.0.0.0.1.enforcement(),
            bcs.0.0.0.1.enforcement(),
            bcs.0.0.1.enforcement(),
            bcs.0.1.enforcement(),
            bcs.1.enforcement(),
        ],
        [
            Enforcement::Elimination,
            Enforcement::Multipliers,
            Enforcement::Elimination,
            Enforcement::Multipliers,
            Enforcement::Elimination,
        ]
    );
}

#[test]
fn a_condition_held_by_multipliers_keeps_its_indices_and_prescription() {
    let held = BoundaryConditions::<Solid<3>>::new()
        .fix_nodes_by(Enforcement::Multipliers, Displacement::<1>, [2, 0, 2])
        .prescribe_node_by(
            Enforcement::Multipliers,
            Displacement::<2>,
            1,
            Quantity::<Length>::new(0.5),
        )
        .prescribe_node(Displacement::<0>, 3, Quantity::<Length>::new(0.75))
        .fix_node(Displacement::<0>, 4);
    let constraints = held.constraint();
    assert_eq!(constraints.len(), 4);
    assert!(matches!(
        &constraints[0],
        EqualityConstraint::Multipliers(entries) if entries == &[
            (7, Prescription::Initial),
            (1, Prescription::Initial),
            (7, Prescription::Initial),
        ]
    ));
    assert!(matches!(
        &constraints[1],
        EqualityConstraint::Multipliers(entries) if entries == &[(5, Prescription::Offset(0.5))]
    ));
    assert!(matches!(
        &constraints[2],
        EqualityConstraint::Prescribed(entries) if entries == &[(9, Prescription::Offset(0.75))]
    ));
    assert!(matches!(
        &constraints[3],
        EqualityConstraint::Fixed(indices) if indices == &[12]
    ));
}

#[test]
fn lowered_the_values_move_the_state() {
    let mut state = Vector::zero(4);
    let constraint = BoundaryConditions::<Solid<2>>::new()
        .prescribe_node(Displacement::<1>, 1, Quantity::<Length>::new(0.5))
        .prescribe_node(Displacement::<0>, 0, Quantity::<Length>::new(-1.0))
        .constraint();
    let constraint = EqualityConstraint::lower(constraint, &mut state);
    assert_eq!(indices(constraint), [0, 3]);
    assert_eq!(state, Vector::from([-1.0, 0.0, 0.0, 0.5]));
}

#[test]
fn a_later_condition_on_a_degree_of_freedom_wins() {
    let mut state = Vector::zero(4);
    let constraint = BoundaryConditions::<Solid<2>>::new()
        .prescribe_node(Displacement::<1>, 1, Quantity::<Length>::new(0.5))
        .prescribe_node(Displacement::<1>, 1, Quantity::<Length>::new(0.75))
        .constraint();
    let constraint = EqualityConstraint::lower(constraint, &mut state);
    assert_eq!(indices(constraint), [3]);
    assert_eq!(state, Vector::from([0.0, 0.0, 0.0, 0.75]));
}
