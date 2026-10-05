use crate::{
    domain::{
        Model,
        qmm::{Discretization, Support, block::Block},
    },
    geometry::mesh::test::tetrahedra,
    math::{Quantity, Tensor},
    units::Density,
};

const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);

fn model() -> Model<Block<(), Quantity<Density>>, 3> {
    let support = |spacing, reach| Support {
        spacing: Quantity::new(spacing),
        reach,
    };
    let discretization =
        Discretization::new(&tetrahedra(8), support(0.4, 2.6), support(0.2, 3.6), 3, 1).unwrap();
    let coordinates = discretization.coordinates().clone();
    (
        Block::from(((), discretization)).with_density(DENSITY),
        coordinates,
    )
        .into()
}

#[test]
fn consistent_masses_sum_to_the_mass_of_the_body() {
    let masses = model().nodal_masses();
    let total: f64 = masses
        .iter()
        .flat_map(|row| row.entries().map(|(_, mass)| mass.value()))
        .sum();
    assert!((total / DENSITY.value() - 1.0).abs() < 1e-12, "{total}");
}

#[test]
fn consistent_masses_are_symmetric() {
    let masses = model().nodal_masses();
    masses.iter().enumerate().for_each(|(a, row)| {
        row.entries().for_each(|(b, mass)| {
            assert!((mass.value() - masses[b][a].value()).abs() < 1e-9 * DENSITY.value());
        })
    });
}
