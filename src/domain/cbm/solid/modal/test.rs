use crate::{
    cbm::{Model, block::Block},
    constitutive::{
        canonical::Canonical, fluid::viscoplastic::ViscoplasticFlow,
        solid::hyperelastic::NeoHookean,
    },
    domain::solid::modal::test::{
        coordinates, density, flowing, kuhn_tets, lumped_bar_tests, material,
        viscoplastic_bar_tests,
    },
    geometry::mesh::PrimitiveConnectivity,
    math::Quantity,
    units::Density,
};

type Particles = Block<NeoHookean, Quantity<Density>>;
type Flowing = Block<Canonical<NeoHookean, ViscoplasticFlow>, Quantity<Density>>;

fn bar() -> Model<Particles, 3> {
    let reference = coordinates().into();
    (
        (
            material(),
            density(),
            PrimitiveConnectivity::from(kuhn_tets()),
            &reference,
        )
            .into(),
        reference,
    )
        .into()
}

fn flowing_bar() -> Model<Flowing, 3> {
    let reference = coordinates().into();
    (
        (
            flowing(),
            density(),
            PrimitiveConnectivity::from(kuhn_tets()),
            &reference,
        )
            .into(),
        reference,
    )
        .into()
}

mod tetrahedra {
    use super::*;
    lumped_bar_tests!(bar, 5e-3);
}

mod viscoplastic {
    use super::*;
    viscoplastic_bar_tests!(flowing_bar, 1, 1.3, 5e-3);
}
