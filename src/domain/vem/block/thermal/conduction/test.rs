use super::super::test::{B, INTERIOR, NODES, block, coordinates, pseudorandom, temperatures};
use crate::{
    EPSILON,
    domain::{Model, thermal::conduction::ThermalConductionElements},
    math::{Hessian, Scalar, Tensor},
    vem::NodalReferenceCoordinates,
};

fn forces(block: &B, values: &[Scalar]) -> Vec<Scalar> {
    block
        .nodal_forces(&temperatures(values))
        .unwrap()
        .iter()
        .map(|force| force.value())
        .collect()
}

fn assert_close(a: Scalar, b: Scalar, tolerance: Scalar) {
    assert!((a - b).abs() <= tolerance, "{a} != {b}");
}

#[test]
fn linear_field_is_in_equilibrium_at_the_interior_node() {
    let gradient = [0.3, -0.7, 1.1];
    let values = coordinates()
        .iter()
        .map(|node| {
            2.0 + (0..3)
                .map(|axis| gradient[axis] * node[axis])
                .sum::<Scalar>()
        })
        .collect::<Vec<_>>();
    let found = forces(&block(), &values);
    assert_close(found[INTERIOR], 0.0, 1e-12);
    assert!(
        found
            .iter()
            .enumerate()
            .any(|(index, force)| index != INTERIOR && force.abs() > 1e-3),
        "the boundary nodes must carry the flux"
    )
}

#[test]
fn forces_sum_to_zero() {
    assert_close(forces(&block(), &pseudorandom()).iter().sum(), 0.0, 1e-11)
}

#[test]
fn forces_are_the_gradient_of_the_potential() {
    let block = block();
    let values = pseudorandom();
    let found = forces(&block, &values);
    (0..NODES).for_each(|node| {
        let shifted = |shift: Scalar| {
            let mut values = values.clone();
            values[node] += shift;
            block.potential(&temperatures(&values)).unwrap().value()
        };
        assert_close(
            found[node],
            (shifted(EPSILON) - shifted(-EPSILON)) / (2.0 * EPSILON),
            1e-7,
        )
    })
}

#[test]
fn stiffnesses_are_the_gradient_of_the_forces() {
    let block = block();
    let values = pseudorandom();
    let found = block.nodal_stiffnesses(&temperatures(&values)).unwrap();
    (0..NODES).for_each(|node| {
        let shifted = |shift: Scalar| {
            let mut values = values.clone();
            values[node] += shift;
            forces(&block, &values)
        };
        let (above, below) = (shifted(EPSILON), shifted(-EPSILON));
        (0..NODES).for_each(|other| {
            let entry = found.entry(other, node);
            assert_close(entry, (above[other] - below[other]) / (2.0 * EPSILON), 1e-7)
        })
    })
}

#[test]
fn model_assembles_the_block() {
    let values = temperatures(&pseudorandom());
    let model: Model<B, 3> = (block(), NodalReferenceCoordinates::from(coordinates())).into();
    model
        .nodal_forces(&values)
        .unwrap()
        .iter()
        .zip(block().nodal_forces(&values).unwrap().iter())
        .for_each(|(a, b)| assert_close(a.value(), b.value(), 0.0))
}
