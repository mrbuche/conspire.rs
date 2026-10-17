use crate::{
    constitutive::electrical::dielectric::Linear,
    domain::diffusion::DiffusionElements,
    fem::{
        Blocks, Minimize, Model, NodalReferenceCoordinates, Root,
        block::{
            Block,
            electrical::{
                NodalPotentials,
                dielectric::{NodalForcesElectrical, NodalStiffnessesElectrical},
            },
            element::linear::Tetrahedron,
        },
        electrical::dielectric::DielectricElements,
    },
    math::{Matrix, Quantity, Tensor, Vector, optimize::NewtonRaphson},
    units::{Energy, Permittivity},
};

const WIDTH: f64 = 0.2;
const DEPTH: f64 = 0.3;
const GAP: f64 = 0.05;
const VOLTAGE: f64 = 100.0;
const NEAR: f64 = 1e-9;

type B = Block<Linear, Tetrahedron<4>, 4, 3, 4, 4>;

const CUBOID: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 5, 1, 7],
    [0, 3, 2, 7],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 6, 4, 7],
];

fn connectivity(offset: usize) -> Vec<[usize; 4]> {
    CUBOID
        .iter()
        .map(|tetrahedron| tetrahedron.map(|node| node + offset))
        .collect()
}

fn coordinates(levels: &[f64]) -> NodalReferenceCoordinates<3> {
    (0..4 * levels.len())
        .map(|node| {
            [
                WIDTH * (node % 2) as f64,
                DEPTH * ((node / 2) % 2) as f64,
                levels[node / 4],
            ]
        })
        .collect::<Vec<_>>()
        .into()
}

fn permittivity(value: f64) -> Linear {
    Linear {
        permittivity: Permittivity::farads_per_meter(value),
    }
}

fn slab() -> (Model<B, 3>, NodalReferenceCoordinates<3>) {
    let coordinates = coordinates(&[0.0, GAP]);
    (
        (
            B::from((permittivity(2.5), connectivity(0), &coordinates)),
            coordinates.clone(),
        )
            .into(),
        coordinates,
    )
}

fn layers() -> (Model<Blocks<B, B>, 3>, NodalReferenceCoordinates<3>, f64) {
    let coordinates = coordinates(&[0.0, GAP, 3.0 * GAP]);
    (
        (
            Blocks(
                B::from((permittivity(2.5), connectivity(0), &coordinates)),
                B::from((permittivity(10.0), connectivity(4), &coordinates)),
            ),
            coordinates.clone(),
        )
            .into(),
        coordinates,
        1.0 / (GAP / (2.5 * WIDTH * DEPTH) + 2.0 * GAP / (10.0 * WIDTH * DEPTH)),
    )
}

fn electrodes(
    coordinates: &NodalReferenceCoordinates<3>,
    top: f64,
) -> (
    crate::fem::boundary::BoundaryConditions<
        'static,
        crate::fem::boundary::Field<crate::units::ElectricPotential>,
    >,
    Vec<usize>,
    Vec<usize>,
) {
    let nodes = coordinates.len();
    let on = |height: f64| {
        (0..nodes)
            .filter(|&node| coordinates[node][2].value() == height)
            .collect::<Vec<_>>()
    };
    let (bottom, top) = (on(0.0), on(top));
    let mut matrix = Matrix::zero(bottom.len() + top.len(), nodes);
    let mut rhs = Vector::zero(bottom.len() + top.len());
    bottom
        .iter()
        .chain(&top)
        .enumerate()
        .for_each(|(row, &node)| matrix[row][node] = 1.0);
    (0..top.len()).for_each(|row| rhs[bottom.len() + row] = VOLTAGE);
    (
        crate::fem::block::test::field_conditions(
            crate::fem::boundary::Potential,
            &matrix,
            &rhs,
            Quantity::<crate::units::ElectricPotential>::new,
        ),
        bottom,
        top,
    )
}

fn near(found: f64, expected: f64) {
    assert!(
        (found - expected).abs() <= NEAR * expected.abs().max(1.0),
        "{found} is not within {NEAR} of {expected}"
    )
}

fn charge(forces: &NodalForcesElectrical, nodes: &[usize]) -> f64 {
    nodes.iter().map(|&node| forces[node].value()).sum()
}

fn solve<Bl>(
    model: &Model<Bl, 3>,
    constraint: crate::fem::boundary::BoundaryConditions<
        '_,
        crate::fem::boundary::Field<crate::units::ElectricPotential>,
    >,
) -> NodalPotentials
where
    Bl: DielectricElements,
{
    <Model<Bl, 3> as Root<NodalForcesElectrical, NodalStiffnessesElectrical, NodalPotentials>>::root(
        model,
        &constraint,
        NewtonRaphson::default(),
    )
    .unwrap()
}

mod parallel_plate {
    use super::*;
    #[test]
    fn the_potential_rises_linearly_across_the_gap() {
        let (model, coordinates) = slab();
        let (constraint, _, _) = electrodes(&coordinates, GAP);
        let potentials = solve(&model, constraint);
        (0..8).for_each(|node| {
            near(
                potentials[node].value(),
                VOLTAGE * coordinates[node][2].value() / GAP,
            )
        })
    }
    #[test]
    fn the_electrodes_hold_equal_and_opposite_charge_of_capacitance_times_voltage() {
        let (model, coordinates) = slab();
        let (constraint, bottom, top) = electrodes(&coordinates, GAP);
        let potentials = solve(&model, constraint);
        let forces = model.nodal_forces(&potentials).unwrap();
        let capacitance = 2.5 * WIDTH * DEPTH / GAP;
        near(charge(&forces, &top), capacitance * VOLTAGE);
        near(charge(&forces, &bottom), -capacitance * VOLTAGE)
    }
    #[test]
    fn the_stored_energy_is_half_capacitance_times_voltage_squared() {
        let (model, coordinates) = slab();
        let (constraint, _, _) = electrodes(&coordinates, GAP);
        let potentials = solve(&model, constraint);
        near(
            model.potential(&potentials).unwrap().value(),
            0.5 * (2.5 * WIDTH * DEPTH / GAP) * VOLTAGE * VOLTAGE,
        )
    }
    #[test]
    fn the_free_nodes_carry_no_charge() {
        let (model, coordinates) = slab();
        let (constraint, bottom, top) = electrodes(&coordinates, GAP);
        let potentials = solve(&model, constraint);
        let forces = model.nodal_forces(&potentials).unwrap();
        (0..8)
            .filter(|node| !bottom.contains(node) && !top.contains(node))
            .for_each(|node| near(forces[node].value(), 0.0))
    }
    #[test]
    fn minimizing_the_energy_gives_the_same_potentials() {
        let (model, coordinates) = slab();
        let (constraint, _, _) = electrodes(&coordinates, GAP);
        let solved = solve(&model, constraint);
        let (constraint, _, _) = electrodes(&coordinates, GAP);
        let minimized = <Model<B, 3> as Minimize<
            Quantity<Energy>,
            NodalForcesElectrical,
            NodalPotentials,
        >>::minimize(&model, &constraint, NewtonRaphson::default())
        .unwrap();
        (0..8).for_each(|node| near(minimized[node].value(), solved[node].value()))
    }
}

mod series {
    use super::*;
    #[test]
    fn two_layers_make_capacitors_in_series() {
        let (model, coordinates, capacitance) = layers();
        let (constraint, bottom, top) = electrodes(&coordinates, 3.0 * GAP);
        let potentials = solve(&model, constraint);
        let forces = model.nodal_forces(&potentials).unwrap();
        near(charge(&forces, &top), capacitance * VOLTAGE);
        near(charge(&forces, &bottom), -capacitance * VOLTAGE)
    }
    #[test]
    fn the_potential_across_each_layer_divides_inversely_with_its_capacitance() {
        let (model, coordinates, capacitance) = layers();
        let (constraint, _, _) = electrodes(&coordinates, 3.0 * GAP);
        let potentials = solve(&model, constraint);
        let middle = (0..coordinates.len())
            .find(|&node| coordinates[node][2].value() == GAP)
            .unwrap();
        near(
            potentials[middle].value(),
            capacitance * VOLTAGE / (2.5 * WIDTH * DEPTH / GAP),
        )
    }
}
