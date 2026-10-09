use crate::{
    cbm::{Model, block::Block},
    domain::NodalReferenceCoordinates,
    geometry::mesh::PrimitiveConnectivity,
    math::{
        Quantity, Tensor,
        assert::{Assert, AssertionError},
    },
    mechanics::ReferenceCoordinate,
    units::{Density, Mass, Volume},
};

const LEFT: Quantity<Density> = Density::kilograms_per_cubic_meter(1e3);
const RIGHT: Quantity<Density> = Density::kilograms_per_cubic_meter(3e3);

const COORDINATES: [[f64; 3]; 5] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
];

fn problem() -> (PrimitiveConnectivity<3, 4>, NodalReferenceCoordinates<3>) {
    (
        PrimitiveConnectivity::from(vec![[0, 1, 2, 3], [1, 4, 2, 3]]),
        NodalReferenceCoordinates::from(COORDINATES),
    )
}

fn field(coordinate: &ReferenceCoordinate) -> Quantity<Density> {
    if coordinate[0].value() < 0.5 {
        LEFT
    } else {
        RIGHT
    }
}

fn volume() -> Quantity<Volume> {
    Volume::cubic_meters(1.0 / 6.0 + 1.0 / 3.0)
}

#[test]
fn a_uniform_density_block_has_the_mass_of_its_volume() -> Result<(), AssertionError> {
    let (connectivity, coordinates) = problem();
    let block = Block::from(((), LEFT, connectivity, &coordinates));
    Assert::default().eq_within_tols(block.mass(), &(LEFT * volume()))
}

#[test]
fn the_nodal_masses_total_the_block_mass() -> Result<(), AssertionError> {
    let (connectivity, coordinates) = problem();
    let block = Block::from(((), field, connectivity, &coordinates));
    let mass = block.mass();
    let model = Model::from((block, coordinates));
    let total: Quantity<Mass> = model.nodal_lumped_masses().iter().copied().sum();
    Assert::default().eq_within_tols(total, &mass)
}

#[test]
fn the_nodal_mass_is_the_density_at_the_particle_times_its_volume() -> Result<(), AssertionError> {
    let (connectivity, coordinates) = problem();
    let block = Block::from(((), field, connectivity, &coordinates));
    let masses = Model::from((block, coordinates)).nodal_lumped_masses();
    let quarter = |first: f64, second: f64| Volume::cubic_meters((first + second) / 4.0);
    let (first, second) = (1.0 / 6.0, 1.0 / 3.0);
    let expected = [
        LEFT * quarter(first, 0.0),
        RIGHT * quarter(first, second),
        LEFT * quarter(first, second),
        LEFT * quarter(first, second),
        RIGHT * quarter(0.0, second),
    ];
    masses
        .iter()
        .zip(expected)
        .try_for_each(|(mass, expected)| Assert::default().eq_within_tols(mass, &expected))
}

#[test]
fn a_field_and_a_uniform_density_agree_when_the_field_is_constant() -> Result<(), AssertionError> {
    let (connectivity, coordinates) = problem();
    let uniform = Block::from(((), LEFT, problem().0, &coordinates));
    let constant = Block::from((
        (),
        |_: &ReferenceCoordinate| LEFT,
        connectivity,
        &coordinates,
    ));
    Assert::default().eq_within_tols(uniform.mass(), &constant.mass())
}
