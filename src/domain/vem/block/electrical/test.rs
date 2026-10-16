use super::super::thermal::test::{
    INTERIOR, MODEL, NODES, block, block_with, coordinates, pseudorandom, temperatures,
};
use crate::{
    EPSILON,
    constitutive::electrical::dielectric::Linear,
    domain::{diffusion::DiffusionElements, electrical::NodalPotentials},
    math::{Hessian, Quantity, Scalar, Tensor},
    units::{ElectricPotential, Permittivity},
    vem::block::{Block, element::Element},
};

type B = Block<Linear, Element>;

fn dielectric() -> B {
    block_with(Linear {
        permittivity: Permittivity::farads_per_meter(2.5),
    })
}

fn potentials(values: &[Scalar]) -> NodalPotentials {
    values
        .iter()
        .map(|&value| Quantity::<ElectricPotential>::new(value))
        .collect()
}

fn charges(block: &B, values: &[Scalar]) -> Vec<Scalar> {
    block
        .nodal_forces(&potentials(values))
        .unwrap()
        .iter()
        .map(|charge| charge.value())
        .collect()
}

fn assert_close(a: Scalar, b: Scalar, tolerance: Scalar) {
    assert!((a - b).abs() <= tolerance, "{a} != {b}");
}

#[test]
fn a_linear_potential_leaves_the_interior_node_uncharged() {
    let gradient = [0.3, -0.7, 1.1];
    let values = coordinates()
        .iter()
        .map(|node| {
            2.0 + (0..3)
                .map(|axis| gradient[axis] * node[axis])
                .sum::<Scalar>()
        })
        .collect::<Vec<_>>();
    let found = charges(&dielectric(), &values);
    assert_close(found[INTERIOR], 0.0, 1e-12);
    assert!(
        found
            .iter()
            .enumerate()
            .any(|(index, charge)| index != INTERIOR && charge.abs() > 1e-3),
        "the boundary nodes must carry the charge"
    )
}

#[test]
fn the_charges_sum_to_zero() {
    assert_close(
        charges(&dielectric(), &pseudorandom()).iter().sum(),
        0.0,
        1e-11,
    )
}

#[test]
fn the_charges_are_the_gradient_of_the_energy() {
    let block = dielectric();
    let values = pseudorandom();
    let found = charges(&block, &values);
    (0..NODES).for_each(|node| {
        let shifted = |shift: Scalar| {
            let mut values = values.clone();
            values[node] += shift;
            block.potential(&potentials(&values)).unwrap().value()
        };
        assert_close(
            found[node],
            (shifted(EPSILON) - shifted(-EPSILON)) / (2.0 * EPSILON),
            1e-7,
        )
    })
}

#[test]
fn the_stiffnesses_are_the_gradient_of_the_charges() {
    let block = dielectric();
    let values = pseudorandom();
    let found = block.nodal_stiffnesses(&potentials(&values)).unwrap();
    (0..NODES).for_each(|node| {
        let shifted = |shift: Scalar| {
            let mut values = values.clone();
            values[node] += shift;
            charges(&block, &values)
        };
        let (above, below) = (shifted(EPSILON), shifted(-EPSILON));
        (0..NODES).for_each(|other| {
            let entry = found.entry(other, node);
            assert_close(entry, (above[other] - below[other]) / (2.0 * EPSILON), 1e-7)
        })
    })
}

#[test]
fn the_charges_match_the_heat_flows_of_a_conductor_with_the_same_number() {
    assert_eq!(MODEL.thermal_conductivity.value(), 2.5);
    let values = pseudorandom();
    let flows = block().nodal_forces(&temperatures(&values)).unwrap();
    charges(&dielectric(), &values)
        .iter()
        .zip(flows.iter())
        .for_each(|(charge, flow)| assert_close(*charge, flow.value(), 1e-12))
}
