use super::BoundaryConditions;
use crate::{
    domain::boundary::{
        Concentration, Displacement, Field, Potential, Solid, SolidThermal, Temperature,
    },
    geometry::mesh::{NodeSet, NodeSets},
    math::optimize::EqualityConstraint,
    units::{Charge, Current, MolarFlowRate, Power},
};

fn indices(constraint: EqualityConstraint) -> Vec<usize> {
    match constraint {
        EqualityConstraint::Fixed(indices) => indices,
        _ => panic!("expected a fixed constraint"),
    }
}

fn sets() -> NodeSets {
    vec![
        NodeSet::new(vec![0, 2]).with_label(7).with_name("bottom"),
        NodeSet::new(vec![1]).with_name("top"),
    ]
    .into()
}

#[test]
fn a_component_of_a_set() {
    let sets = sets();
    let fixed = BoundaryConditions::<Solid<3>>::new()
        .fix::<Displacement<1>>(sets.by_name("bottom").unwrap());
    assert_eq!(indices(fixed.constraint()), [1, 7]);
}

#[test]
fn a_set_found_by_its_label() {
    let sets = sets();
    let fixed =
        BoundaryConditions::<Solid<3>>::new().fix::<Displacement<1>>(sets.by_label(7).unwrap());
    assert_eq!(indices(fixed.constraint()), [1, 7]);
}

#[test]
fn a_solid_in_two_dimensions() {
    let sets = sets();
    let fixed =
        BoundaryConditions::<Solid<2>>::new().fix::<Displacement<1>>(sets.by_name("top").unwrap());
    assert_eq!(indices(fixed.constraint()), [3]);
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
    let fixed = BoundaryConditions::<SolidThermal<3>>::new().fix_nodes::<Temperature>(0..3);
    assert_eq!(indices(fixed.constraint()), [3, 7, 11]);
}

#[test]
fn a_solid_and_its_temperature() {
    let fixed = BoundaryConditions::<SolidThermal<3>>::new()
        .fix_nodes::<Temperature>(0..5)
        .fix_node::<Displacement<0>>(0)
        .fix_node::<Displacement<1>>(0)
        .fix_node::<Displacement<2>>(0)
        .fix_node::<Displacement<1>>(1)
        .fix_node::<Displacement<2>>(1)
        .fix_node::<Displacement<2>>(2);
    let mut expected: Vec<usize> = (0..5)
        .map(|node| 4 * node + 3)
        .chain([0, 1, 2, 5, 6, 10])
        .collect();
    expected.sort_unstable();
    assert_eq!(indices(fixed.constraint()), expected);
}

#[test]
fn a_scalar_field_is_indexed_by_its_node() {
    let sets = sets();
    let bottom = sets.by_name("bottom").unwrap();
    let temperature = BoundaryConditions::<Field<Power>>::new().fix::<Temperature>(bottom);
    let potential = BoundaryConditions::<Field<Charge>>::new().fix::<Potential>(bottom);
    let current = BoundaryConditions::<Field<Current>>::new().fix::<Potential>(bottom);
    let concentration =
        BoundaryConditions::<Field<MolarFlowRate>>::new().fix::<Concentration>(bottom);
    assert_eq!(indices(temperature.constraint()), [0, 2]);
    assert_eq!(indices(potential.constraint()), [0, 2]);
    assert_eq!(indices(current.constraint()), [0, 2]);
    assert_eq!(indices(concentration.constraint()), [0, 2]);
}

#[test]
fn repeats_are_held_once() {
    let sets = sets();
    let fixed = BoundaryConditions::<Solid<3>>::new()
        .fix_nodes::<Displacement<0>>([1, 1, 2])
        .fix_node::<Displacement<0>>(1)
        .fix::<Displacement<0>>(sets.by_name("bottom").unwrap());
    assert_eq!(indices(fixed.constraint()), [0, 3, 6]);
}

#[test]
fn nothing_fixed() {
    let fixed = BoundaryConditions::<Solid<3>>::new();
    assert!(indices(fixed.constraint()).is_empty());
}

#[test]
fn the_conditions_are_kept_in_order() {
    let sets = sets();
    let fixed = BoundaryConditions::<Solid<3>>::new()
        .fix::<Displacement<1>>(sets.by_name("top").unwrap())
        .fix_node::<Displacement<2>>(4);
    let conditions = fixed.conditions();
    assert_eq!(conditions.len(), 2);
    assert_eq!(conditions[0].indices(), [4]);
    assert_eq!(conditions[1].indices(), [14]);
}

#[test]
fn a_condition_follows_the_set_it_borrows() {
    let sets = sets();
    let bottom = sets.by_name("bottom").unwrap();
    let before = BoundaryConditions::<Solid<3>>::new().fix::<Displacement<0>>(bottom);
    let other = NodeSet::new(vec![5]);
    let after = BoundaryConditions::<Solid<3>>::new().fix::<Displacement<0>>(&other);
    assert_eq!(indices(before.constraint()), [0, 6]);
    assert_eq!(indices(after.constraint()), [15]);
}

#[test]
fn a_set_that_is_not_there() {
    assert!(sets().by_name("side").is_none());
    assert!(sets().by_label(8).is_none());
}
