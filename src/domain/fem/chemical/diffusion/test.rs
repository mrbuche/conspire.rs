use crate::{
    constitutive::chemical::diffusion::Fick,
    domain::diffusion::DiffusionElements,
    fem::{
        Blocks, Minimize, Model, NodalReferenceCoordinates, Root,
        block::{Block, element::linear::Tetrahedron},
        chemical::diffusion::{
            ChemicalDiffusionElements, NodalConcentrations, NodalMolarFlowRates,
            NodalVolumetricFlowRates,
        },
    },
    math::{
        Matrix, Quantity, Tensor, Vector,
        optimize::{EqualityConstraint, NewtonRaphson},
    },
    units::{Diffusivity, MolarFlowRateConcentration},
};

const WIDTH: f64 = 0.2;
const DEPTH: f64 = 0.3;
const GAP: f64 = 0.05;
const DELTA: f64 = 100.0;
const NEAR: f64 = 1e-9;

type B = Block<Fick, Tetrahedron<4>, 4, 3, 4, 4>;

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

fn diffusivity(value: f64) -> Fick {
    Fick {
        diffusivity: Diffusivity::square_meters_per_second(value),
    }
}

fn slab() -> (Model<B, 3>, NodalReferenceCoordinates<3>) {
    let coordinates = coordinates(&[0.0, GAP]);
    (
        (
            B::from((diffusivity(2.5), connectivity(0), &coordinates)),
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
                B::from((diffusivity(2.5), connectivity(0), &coordinates)),
                B::from((diffusivity(10.0), connectivity(4), &coordinates)),
            ),
            coordinates.clone(),
        )
            .into(),
        coordinates,
        1.0 / (GAP / (2.5 * WIDTH * DEPTH) + 2.0 * GAP / (10.0 * WIDTH * DEPTH)),
    )
}

fn ends(
    coordinates: &NodalReferenceCoordinates<3>,
    top: f64,
) -> (EqualityConstraint, Vec<usize>, Vec<usize>) {
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
    (0..top.len()).for_each(|row| rhs[bottom.len() + row] = DELTA);
    (EqualityConstraint::Linear(matrix, rhs), bottom, top)
}

fn near(found: f64, expected: f64) {
    assert!(
        (found - expected).abs() <= NEAR * expected.abs().max(1.0),
        "{found} is not within {NEAR} of {expected}"
    )
}

fn molar_flow(forces: &NodalMolarFlowRates, nodes: &[usize]) -> f64 {
    nodes.iter().map(|&node| forces[node].value()).sum()
}

fn solve<Bl>(model: &Model<Bl, 3>, constraint: EqualityConstraint) -> NodalConcentrations
where
    Bl: ChemicalDiffusionElements,
{
    <Model<Bl, 3> as Root<NodalMolarFlowRates, NodalVolumetricFlowRates, NodalConcentrations>>::root(
        model,
        &vec![constraint].into(),
        NewtonRaphson::default(),
    )
    .unwrap()
}

mod bar {
    use super::*;
    #[test]
    fn the_concentration_rises_linearly_along_the_bar() {
        let (model, coordinates) = slab();
        let (constraint, _, _) = ends(&coordinates, GAP);
        let potentials = solve(&model, constraint);
        (0..8).for_each(|node| {
            near(
                potentials[node].value(),
                DELTA * coordinates[node][2].value() / GAP,
            )
        })
    }
    #[test]
    fn the_ends_carry_equal_and_opposite_molar_flow_of_transfer_times_the_concentration_difference()
    {
        let (model, coordinates) = slab();
        let (constraint, bottom, top) = ends(&coordinates, GAP);
        let potentials = solve(&model, constraint);
        let forces = model.nodal_forces(&potentials).unwrap();
        let transfer = 2.5 * WIDTH * DEPTH / GAP;
        near(molar_flow(&forces, &top), transfer * DELTA);
        near(molar_flow(&forces, &bottom), -transfer * DELTA)
    }
    #[test]
    fn the_potential_is_half_the_transfer_times_the_concentration_difference_squared() {
        let (model, coordinates) = slab();
        let (constraint, _, _) = ends(&coordinates, GAP);
        let potentials = solve(&model, constraint);
        near(
            model.potential(&potentials).unwrap().value(),
            0.5 * (2.5 * WIDTH * DEPTH / GAP) * DELTA * DELTA,
        )
    }
    #[test]
    fn the_free_nodes_carry_no_molar_flow() {
        let (model, coordinates) = slab();
        let (constraint, bottom, top) = ends(&coordinates, GAP);
        let potentials = solve(&model, constraint);
        let forces = model.nodal_forces(&potentials).unwrap();
        (0..8)
            .filter(|node| !bottom.contains(node) && !top.contains(node))
            .for_each(|node| near(forces[node].value(), 0.0))
    }
    #[test]
    fn minimizing_the_energy_gives_the_same_potentials() {
        let (model, coordinates) = slab();
        let (constraint, _, _) = ends(&coordinates, GAP);
        let solved = solve(&model, constraint);
        let (constraint, _, _) = ends(&coordinates, GAP);
        let minimized =
            <Model<B, 3> as Minimize<
                Quantity<MolarFlowRateConcentration>,
                NodalMolarFlowRates,
                NodalConcentrations,
            >>::minimize(&model, &vec![constraint].into(), NewtonRaphson::default())
            .unwrap();
        (0..8).for_each(|node| near(minimized[node].value(), solved[node].value()))
    }
}

mod series {
    use super::*;
    #[test]
    fn two_layers_make_transfers_in_series() {
        let (model, coordinates, transfer) = layers();
        let (constraint, bottom, top) = ends(&coordinates, 3.0 * GAP);
        let potentials = solve(&model, constraint);
        let forces = model.nodal_forces(&potentials).unwrap();
        near(molar_flow(&forces, &top), transfer * DELTA);
        near(molar_flow(&forces, &bottom), -transfer * DELTA)
    }
    #[test]
    fn the_concentration_across_each_layer_divides_inversely_with_its_transfer() {
        let (model, coordinates, transfer) = layers();
        let (constraint, _, _) = ends(&coordinates, 3.0 * GAP);
        let potentials = solve(&model, constraint);
        let middle = (0..coordinates.len())
            .find(|&node| coordinates[node][2].value() == GAP)
            .unwrap();
        near(
            potentials[middle].value(),
            transfer * DELTA / (2.5 * WIDTH * DEPTH / GAP),
        )
    }
}
