use super::BoundaryConditions;
use crate::{
    domain::boundary::{
        Concentration, Displacement, Field, Potential, Solid, SolidThermal, Temperature,
    },
    geometry::mesh::NodeSet,
    math::optimize::EqualityConstraint,
    units::{Charge, Current, MolarFlowRate, Power},
};

fn indices(constraint: EqualityConstraint) -> Vec<usize> {
    match constraint {
        EqualityConstraint::Fixed(indices) => indices,
        _ => panic!("expected a fixed constraint"),
    }
}

#[test]
fn a_component_of_a_solid() {
    let nodes = NodeSet::new(vec![0, 2]);
    let fixed = BoundaryConditions::<Solid<3>>::new().fix::<Displacement<1>>(&nodes);
    assert_eq!(indices(fixed.constraint()), [1, 7]);
}

#[test]
fn a_solid_in_two_dimensions() {
    let nodes = NodeSet::new(vec![3]);
    let fixed = BoundaryConditions::<Solid<2>>::new().fix::<Displacement<1>>(&nodes);
    assert_eq!(indices(fixed.constraint()), [7]);
}

#[test]
fn every_component_of_a_node() {
    let fixed = BoundaryConditions::<Solid<3>>::new()
        .fix_node::<Displacement<0>>(1)
        .fix_node::<Displacement<1>>(1)
        .fix_node::<Displacement<2>>(1);
    assert_eq!(indices(fixed.constraint()), [3, 4, 5]);
}

#[test]
fn temperatures_come_after_the_coordinates() {
    let nodes = NodeSet::new(vec![0, 1, 2]);
    let fixed = BoundaryConditions::<SolidThermal<3>>::new().fix::<Temperature>(&nodes);
    assert_eq!(indices(fixed.constraint()), [3, 7, 11]);
}

#[test]
fn a_solid_and_its_temperature() {
    let nodes = NodeSet::new((0..5).collect());
    let fixed = BoundaryConditions::<SolidThermal<3>>::new()
        .fix::<Temperature>(&nodes)
        .fix_node::<Displacement<0>>(0)
        .fix_node::<Displacement<1>>(0)
        .fix_node::<Displacement<2>>(0)
        .fix_node::<Displacement<1>>(1)
        .fix_node::<Displacement<2>>(1)
        .fix_node::<Displacement<2>>(2);
    let by_hand: Vec<usize> = (0..5)
        .map(|node| 4 * node + 3)
        .chain([0, 1, 2, 5, 6, 10])
        .collect();
    let mut expected = by_hand;
    expected.sort_unstable();
    assert_eq!(indices(fixed.constraint()), expected);
}

#[test]
fn a_scalar_field_is_indexed_by_its_node() {
    let nodes = NodeSet::new(vec![4, 1]);
    let temperature = BoundaryConditions::<Field<Power>>::new().fix::<Temperature>(&nodes);
    let potential = BoundaryConditions::<Field<Charge>>::new().fix::<Potential>(&nodes);
    let current = BoundaryConditions::<Field<Current>>::new().fix::<Potential>(&nodes);
    let concentration =
        BoundaryConditions::<Field<MolarFlowRate>>::new().fix::<Concentration>(&nodes);
    assert_eq!(indices(temperature.constraint()), [1, 4]);
    assert_eq!(indices(potential.constraint()), [1, 4]);
    assert_eq!(indices(current.constraint()), [1, 4]);
    assert_eq!(indices(concentration.constraint()), [1, 4]);
}

#[test]
fn repeats_are_held_once() {
    let nodes = NodeSet::new(vec![1, 1, 2]);
    let fixed = BoundaryConditions::<Solid<3>>::new()
        .fix::<Displacement<0>>(&nodes)
        .fix_node::<Displacement<0>>(1);
    assert_eq!(indices(fixed.constraint()), [3, 6]);
}

#[test]
fn nothing_fixed() {
    assert!(indices(BoundaryConditions::<Solid<3>>::new().constraint()).is_empty());
}

#[test]
fn the_constraint_can_be_made_again() {
    let fixed = BoundaryConditions::<Solid<3>>::new().fix_node::<Displacement<2>>(0);
    assert_eq!(indices(fixed.constraint()), indices(fixed.constraint()));
}
