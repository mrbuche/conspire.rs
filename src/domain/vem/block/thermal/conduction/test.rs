use crate::{
    EPSILON,
    constitutive::thermal::conduction::Fourier,
    domain::{Model, thermal::conduction::ThermalConductionElements},
    math::{Hessian, Quantity, Scalar, Tensor},
    units::{PowerPerLengthTemperature, Temperature},
    vem::{
        NodalReferenceCoordinates,
        block::{
            Block,
            element::{DEFAULT_STABILIZATION, Element},
            thermal::NodalTemperatures,
        },
    },
};

const CELLS: usize = 2;
const NODES: usize = (CELLS + 1) * (CELLS + 1) * (CELLS + 1);
const INTERIOR: usize = 13;
const MODEL: Fourier = Fourier {
    thermal_conductivity: PowerPerLengthTemperature::watts_per_meter_kelvin(2.5),
};

type B = Block<Fourier, Element>;

fn node(i: usize, j: usize, k: usize) -> usize {
    i + (CELLS + 1) * (j + (CELLS + 1) * k)
}

fn coordinates() -> Vec<[Scalar; 3]> {
    (0..NODES)
        .map(|index| {
            let (i, j, k) = (
                index % (CELLS + 1),
                (index / (CELLS + 1)) % (CELLS + 1),
                index / ((CELLS + 1) * (CELLS + 1)),
            );
            let bump = |axis: usize| ((index * 3 + axis) as Scalar * 5.3).sin() * 0.12;
            [
                i as Scalar + bump(0),
                j as Scalar + bump(1),
                k as Scalar + bump(2),
            ]
        })
        .collect()
}

fn block() -> B {
    let corners = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ];
    let local_faces = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [3, 7, 6, 2],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
    ];
    let mut elements_faces = Vec::new();
    let mut faces_nodes = Vec::new();
    (0..CELLS).for_each(|k| {
        (0..CELLS).for_each(|j| {
            (0..CELLS).for_each(|i| {
                let global = corners.map(|(a, b, c)| node(i + a, j + b, k + c));
                elements_faces.push(
                    local_faces
                        .iter()
                        .map(|face| {
                            faces_nodes.push(face.map(|corner| global[corner]).to_vec());
                            faces_nodes.len() - 1
                        })
                        .collect(),
                )
            })
        })
    });
    (
        MODEL,
        elements_faces,
        faces_nodes,
        &NodalReferenceCoordinates::from(coordinates()),
        DEFAULT_STABILIZATION,
    )
        .into()
}

fn temperatures(values: &[Scalar]) -> NodalTemperatures {
    values
        .iter()
        .map(|&value| Quantity::<Temperature>::new(value))
        .collect()
}

fn pseudorandom() -> Vec<Scalar> {
    (0..NODES)
        .map(|index| ((index as Scalar + 1.0) * 12.9898).sin() * 0.5 + 0.7)
        .collect()
}

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
