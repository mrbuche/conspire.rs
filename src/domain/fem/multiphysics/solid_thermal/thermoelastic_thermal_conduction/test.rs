use crate::{
    constitutive::{
        multiphysics::Canonical,
        solid::thermoelastic::{
            AlmansiHamel,
            test::{
                BULK_MODULUS, COEFFICIENT_OF_THERMAL_EXPANSION, REFERENCE_TEMPERATURE,
                SHEAR_MODULUS,
            },
        },
        thermal::conduction::{Fourier, test::THERMAL_CONDUCTIVITY},
    },
    fem::{
        Model, NodalReferenceCoordinates, Root,
        block::{Block, element::linear::Tetrahedron},
        multiphysics::solid_thermal::{
            NodalForcesSolidThermal, NodalStatesSolidThermal, NodalStiffnessesSolidThermal,
        },
    },
    math::{
        Matrix, Vector,
        optimize::{EqualityConstraint, NewtonRaphson},
    },
};

const NODES: usize = 5;
const CONNECTIVITY: [[usize; 4]; 2] = [[0, 1, 2, 3], [1, 2, 3, 4]];
const HOT: f64 = 100.02;

type B = Block<Canonical<AlmansiHamel, Fourier>, Tetrahedron<4>, 4, 3, 4, 4>;

fn coordinates() -> NodalReferenceCoordinates<3> {
    NodalReferenceCoordinates::from([
        [0.1, 0.2, 0.0],
        [1.3, 0.1, 0.2],
        [0.2, 0.9, 0.1],
        [0.3, 0.4, 1.2],
        [1.5, 1.4, 1.3],
    ])
}

fn model() -> Model<B, 3> {
    (
        B::from((
            Canonical::from((
                AlmansiHamel {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
                    reference_temperature: REFERENCE_TEMPERATURE,
                },
                Fourier {
                    thermal_conductivity: THERMAL_CONDUCTIVITY,
                },
            )),
            CONNECTIVITY.to_vec(),
            &coordinates(),
        )),
        coordinates(),
    )
        .into()
}

fn dilation() -> f64 {
    let temperature_change = HOT - REFERENCE_TEMPERATURE.value();
    (1.0 - 2.0 * COEFFICIENT_OF_THERMAL_EXPANSION.value() * temperature_change).powf(-0.5)
}

fn constraint() -> EqualityConstraint {
    let dilation = dilation();
    let mut matrix = Matrix::zero(11, 4 * NODES);
    let mut rhs = Vector::zero(11);
    (0..NODES).for_each(|node| {
        matrix[node][4 * node + 3] = 1.0;
        rhs[node] = HOT
    });
    [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        .iter()
        .enumerate()
        .for_each(|(row, &(node, direction))| {
            matrix[NODES + row][4 * node + direction] = 1.0;
            rhs[NODES + row] = dilation * coordinates()[node][direction].value()
        });
    EqualityConstraint::Linear(matrix, rhs)
}

fn solve() -> NodalStatesSolidThermal {
    <Model<B, 3> as Root<
        NodalForcesSolidThermal,
        NodalStiffnessesSolidThermal,
        NodalStatesSolidThermal,
    >>::root(&model(), constraint(), NewtonRaphson::default())
    .unwrap()
}

#[test]
fn a_uniform_temperature_dilates_the_body_without_stress() {
    let dilation = dilation();
    let states = solve();
    (0..NODES).for_each(|node| {
        (0..3).for_each(|direction| {
            let expected = dilation * coordinates()[node][direction].value();
            let found = states[node].0[direction].value();
            assert!(
                (found - expected).abs() < 1e-8,
                "node {node} direction {direction}: {found} != {expected}"
            )
        })
    })
}

#[test]
fn the_imposed_temperature_is_reached_at_every_node() {
    let states = solve();
    (0..NODES).for_each(|node| {
        let found = states[node].1.value();
        assert!((found - HOT).abs() < 1e-8, "node {node}: {found} != {HOT}")
    })
}
