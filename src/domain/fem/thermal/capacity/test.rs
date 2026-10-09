use crate::{
    fem::{
        Model, NodalReferenceCoordinates,
        block::{
            Block, ElementDensities, element::linear::Tetrahedron,
            thermal::capacity::ElementHeatCapacities,
        },
    },
    math::{
        Quantity, QuantityVector, Tensor,
        assert::{Assert, AssertionError},
    },
    mechanics::ReferenceCoordinate,
    units::{Density, Energy, HeatCapacity, Power, SpecificHeat, Time, VolumetricHeatCapacity},
};

const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);
const SPECIFIC_HEAT: Quantity<SpecificHeat> = SpecificHeat::joules_per_kilogram_kelvin(4.5e2);

fn capacity() -> Quantity<VolumetricHeatCapacity> {
    DENSITY * SPECIFIC_HEAT
}

fn coordinates() -> NodalReferenceCoordinates<3> {
    [
        [0.1, 0.2, 0.0],
        [1.3, 0.1, 0.2],
        [0.2, 0.9, 0.1],
        [0.3, 0.4, 1.2],
        [1.5, 1.4, 1.3],
    ]
    .into()
}

const CONNECTIVITY: [[usize; 4]; 2] = [[0, 1, 2, 3], [1, 2, 3, 4]];

fn total(capacities: impl Iterator<Item = Quantity<HeatCapacity>>) -> Quantity<HeatCapacity> {
    capacities.sum()
}

macro_rules! model {
    ($name:ident, $g:literal) => {
        fn $name()
        -> Model<Block<(), Tetrahedron<$g>, $g, 3, 4, 4, Quantity<VolumetricHeatCapacity>>, 3> {
            (
                Block::<(), Tetrahedron<$g>, $g, 3, 4, 4>::from((
                    (),
                    CONNECTIVITY.to_vec(),
                    &coordinates(),
                ))
                .with_heat_capacity(capacity()),
                coordinates(),
            )
                .into()
        }
    };
}

model!(model_one_point, 1);
model!(model_four_points, 4);

#[test]
fn lumped_conserves_the_heat_capacity() -> Result<(), AssertionError> {
    let model = model_one_point();
    let expected = capacity() * model.blocks.volume();
    Assert::default().eq_within_tols(
        total(model.nodal_lumped_heat_capacities().iter().copied()),
        &expected,
    )
}

#[test]
fn consistent_conserves_the_heat_capacity() -> Result<(), AssertionError> {
    let model = model_four_points();
    let expected = capacity() * model.blocks.volume();
    let sum = total(
        model
            .nodal_heat_capacities()
            .iter()
            .flat_map(|row| row.entries().map(|(_, entry)| *entry)),
    );
    Assert::default().eq_within_tols(sum, &expected)
}

#[test]
fn consistent_lumps_to_its_row_sums() -> Result<(), AssertionError> {
    let capacities = model_four_points().nodal_heat_capacities();
    let lumped = model_four_points().nodal_lumped_heat_capacities();
    capacities
        .iter()
        .zip(lumped.iter())
        .try_for_each(|(row, lumped)| {
            Assert::default().eq_within_tols(total(row.entries().map(|(_, entry)| *entry)), lumped)
        })
}

#[test]
fn varies_with_the_density() -> Result<(), AssertionError> {
    let block = Block::<(), Tetrahedron<4>, 4, 3, 4, 4, ElementDensities<4>>::from((
        (),
        |coordinate: &ReferenceCoordinate| {
            Density::kilograms_per_cubic_meter(1e3 + 5e2 * coordinate[0].value())
        },
        CONNECTIVITY.to_vec(),
        &coordinates(),
    ));
    let expected = block.mass() * SPECIFIC_HEAT;
    let capacities = block
        .density()
        .iter()
        .map(|densities| {
            densities
                .iter()
                .map(|&density| density * SPECIFIC_HEAT)
                .collect()
        })
        .collect::<ElementHeatCapacities<4>>();
    let model = Model::from((block.with_heat_capacity(capacities), coordinates()));
    Assert::default().eq_within_tols(
        total(model.nodal_lumped_heat_capacities().iter().copied()),
        &expected,
    )
}

#[test]
fn factors_to_the_inverse_with_fixed_nodes() {
    let capacities = model_four_points().nodal_heat_capacities();
    let factors = capacities.factor(&[0]).expect("factors");
    let heating = QuantityVector::<Power>::from([
        (Energy::joules(0.0) / Time::seconds(1.0)),
        (Energy::joules(1.0) / Time::seconds(1.0)),
        (Energy::joules(-2.0) / Time::seconds(1.0)),
        (Energy::joules(0.5) / Time::seconds(1.0)),
        (Energy::joules(3.0) / Time::seconds(1.0)),
    ]);
    let rates = factors.nodal_temperature_rates(&heating);
    assert_eq!(rates[0], 0.0);
    (1..5).for_each(|a| {
        let applied: f64 = capacities[a]
            .entries()
            .map(|(b, entry)| entry.value() * rates[b])
            .sum();
        let expected = heating[a].value();
        assert!((applied - expected).abs() <= 1e-9 * expected.abs().max(1.0));
    })
}
