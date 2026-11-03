use crate::{
    cbm::{Model, block::Block},
    constitutive::solid::hyperelastic::NeoHookean,
    domain::solid::modal::test::{coordinates, density, kuhn_tets, lumped_bar_tests, material},
    geometry::mesh::PrimitiveConnectivity,
    math::Quantity,
    units::Density,
};

type Particles = Block<NeoHookean, Quantity<Density>>;

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

mod tetrahedra {
    use super::*;
    lumped_bar_tests!(bar, 5e-3);
}
