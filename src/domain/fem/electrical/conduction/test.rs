use crate::{
    constitutive::electrical::conduction::Ohmic,
    domain::diffusion::DiffusionElements,
    fem::{
        Blocks, Minimize, Model, NodalReferenceCoordinates, Root,
        block::{Block, electrical::NodalPotentials, element::linear::Tetrahedron},
        electrical::conduction::{ElectricalConductionElements, NodalConductances, NodalCurrents},
    },
    math::{Matrix, Quantity, Tensor, Vector, optimize::NewtonRaphson},
    units::{ElectricalConductivity, Power},
};

const WIDTH: f64 = 0.2;
const DEPTH: f64 = 0.3;
const GAP: f64 = 0.05;
const VOLTAGE: f64 = 100.0;
const NEAR: f64 = 1e-9;

type B = Block<Ohmic, Tetrahedron<4>, 4, 3, 4, 4>;

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

fn electrical_conductivity(value: f64) -> Ohmic {
    Ohmic {
        electrical_conductivity: ElectricalConductivity::siemens_per_meter(value),
    }
}

fn slab() -> (Model<B, 3>, NodalReferenceCoordinates<3>) {
    let coordinates = coordinates(&[0.0, GAP]);
    (
        (
            B::from((electrical_conductivity(2.5), connectivity(0), &coordinates)),
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
                B::from((electrical_conductivity(2.5), connectivity(0), &coordinates)),
                B::from((electrical_conductivity(10.0), connectivity(4), &coordinates)),
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
    impl crate::fem::boundary::Constrain<crate::fem::boundary::Field<crate::units::ElectricPotential>>,
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
    crate::math::assert::Assert {
        abs_tol: NEAR * expected.abs().max(1.0),
        rel_tol: NEAR,
        ..Default::default()
    }
    .eq_within_tols(found, &expected)
    .unwrap()
}

fn current(forces: &NodalCurrents, nodes: &[usize]) -> f64 {
    nodes.iter().map(|&node| forces[node].value()).sum()
}

fn solve<Bl>(
    model: &Model<Bl, 3>,
    constraint: impl crate::fem::boundary::Constrain<
        crate::fem::boundary::Field<crate::units::ElectricPotential>,
    >,
) -> NodalPotentials
where
    Bl: ElectricalConductionElements,
{
    <Model<Bl, 3> as Root<NodalCurrents, NodalConductances, NodalPotentials>>::root(
        model,
        &constraint,
        &(),
        NewtonRaphson::default(),
    )
    .unwrap()
}

mod bar {
    use super::*;
    #[test]
    fn the_potential_rises_linearly_along_the_bar() {
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
    fn the_electrodes_carry_equal_and_opposite_current_of_conductance_times_voltage() {
        let (model, coordinates) = slab();
        let (constraint, bottom, top) = electrodes(&coordinates, GAP);
        let potentials = solve(&model, constraint);
        let forces = model.nodal_forces(&potentials).unwrap();
        let conductance = 2.5 * WIDTH * DEPTH / GAP;
        near(current(&forces, &top), conductance * VOLTAGE);
        near(current(&forces, &bottom), -conductance * VOLTAGE)
    }
    #[test]
    fn the_potential_is_half_the_dissipated_power_of_conductance_times_voltage_squared() {
        let (model, coordinates) = slab();
        let (constraint, _, _) = electrodes(&coordinates, GAP);
        let potentials = solve(&model, constraint);
        near(
            model.potential(&potentials).unwrap().value(),
            0.5 * (2.5 * WIDTH * DEPTH / GAP) * VOLTAGE * VOLTAGE,
        )
    }
    #[test]
    fn the_free_nodes_carry_no_current() {
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
            Quantity<Power>,
            NodalCurrents,
            NodalPotentials,
        >>::minimize(&model, &constraint, &(), NewtonRaphson::default())
        .unwrap();
        (0..8).for_each(|node| near(minimized[node].value(), solved[node].value()))
    }
}

mod series {
    use super::*;
    #[test]
    fn two_layers_make_resistors_in_series() {
        let (model, coordinates, conductance) = layers();
        let (constraint, bottom, top) = electrodes(&coordinates, 3.0 * GAP);
        let potentials = solve(&model, constraint);
        let forces = model.nodal_forces(&potentials).unwrap();
        near(current(&forces, &top), conductance * VOLTAGE);
        near(current(&forces, &bottom), -conductance * VOLTAGE)
    }
    #[test]
    fn the_potential_across_each_layer_divides_with_its_resistance() {
        let (model, coordinates, conductance) = layers();
        let (constraint, _, _) = electrodes(&coordinates, 3.0 * GAP);
        let potentials = solve(&model, constraint);
        let middle = (0..coordinates.len())
            .find(|&node| coordinates[node][2].value() == GAP)
            .unwrap();
        near(
            potentials[middle].value(),
            conductance * VOLTAGE / (2.5 * WIDTH * DEPTH / GAP),
        )
    }
}

mod source {
    use super::*;
    use crate::{
        fem::{
            boundary::{BoundaryConditions, Constrain, ConstrainAt as _, Field, Loads, Potential},
            diffusion::NodalSource,
        },
        units::{Current, ElectricPotential},
    };
    const RISE: f64 = 0.5;
    fn linear(coordinates: &NodalReferenceCoordinates<3>) -> NodalPotentials {
        (0..8)
            .map(|node| {
                Quantity::<ElectricPotential>::new(RISE * coordinates[node][2].value() / GAP)
            })
            .collect()
    }
    fn injection(
        model: &Model<B, 3>,
        coordinates: &NodalReferenceCoordinates<3>,
    ) -> Loads<Field<ElectricPotential>, ((), NodalSource<Current>)> {
        let forces = model.nodal_forces(&linear(coordinates)).unwrap();
        let sources = (0..8)
            .map(|node| {
                if node >= 4 {
                    forces[node]
                } else {
                    Quantity::<Current>::new(0.0)
                }
            })
            .collect::<NodalCurrents>();
        Loads::<Field<ElectricPotential>>::new().source(sources)
    }
    fn grounded() -> impl Constrain<Field<ElectricPotential>> {
        BoundaryConditions::new().fix_nodes(Potential, 0..4)
    }
    #[test]
    fn the_currents_a_linear_potential_needs_at_the_top_of_a_slab_give_that_potential() {
        let (model, coordinates) = slab();
        let potentials =
            <Model<B, 3> as Root<NodalCurrents, NodalConductances, NodalPotentials>>::root(
                &model,
                &grounded(),
                &injection(&model, &coordinates),
                NewtonRaphson::default(),
            )
            .unwrap();
        let expected = linear(&coordinates);
        (0..8).for_each(|node| near(potentials[node].value(), expected[node].value()))
    }
    #[test]
    fn the_injected_current_is_the_conductance_times_the_rise() {
        let (model, coordinates) = slab();
        let top = (4..8).collect::<Vec<_>>();
        let sources = model.nodal_forces(&linear(&coordinates)).unwrap();
        near(current(&sources, &top), 2.5 * WIDTH * DEPTH / GAP * RISE)
    }
    #[test]
    fn minimizing_the_energy_with_the_source_gives_the_same_potentials() {
        let (model, coordinates) = slab();
        let solved =
            <Model<B, 3> as Root<NodalCurrents, NodalConductances, NodalPotentials>>::root(
                &model,
                &grounded(),
                &injection(&model, &coordinates),
                NewtonRaphson::default(),
            )
            .unwrap();
        let minimized =
            <Model<B, 3> as Minimize<Quantity<Power>, NodalCurrents, NodalPotentials>>::minimize(
                &model,
                &grounded(),
                &injection(&model, &coordinates),
                NewtonRaphson::default(),
            )
            .unwrap();
        (0..8).for_each(|node| near(minimized[node].value(), solved[node].value()))
    }
}
