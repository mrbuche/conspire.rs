use crate::{
    cbm::{Model, block::Block},
    constitutive::{
        canonical::Canonical,
        fluid::{
            hyperviscous::Newtonian,
            plastic::{Linear, PlasticFlow, VonMises},
            viscoplastic::ViscoplasticFlow,
        },
        solid::hyperelastic::NeoHookean,
    },
    domain::solid::modal::test::{
        coordinates, density, flowing, kuhn_tets, lumped_bar_tests, material, plastic_bar_tests,
        plasticity, viscoelastic_bar_tests, viscoelasticity, viscoplastic_bar_tests,
    },
    geometry::mesh::PrimitiveConnectivity,
    math::Quantity,
    units::Density,
};

type Particles = Block<NeoHookean, Quantity<Density>>;
type Flowing = Block<Canonical<NeoHookean, ViscoplasticFlow>, Quantity<Density>>;
type Plastic = Block<Canonical<NeoHookean, PlasticFlow<VonMises, Linear>>, Quantity<Density>>;
type Viscoelastic = Block<Canonical<NeoHookean, Newtonian>, Quantity<Density>>;

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

fn viscoelastic_bar() -> Model<Viscoelastic, 3> {
    let reference = coordinates().into();
    (
        (
            viscoelasticity(),
            density(),
            PrimitiveConnectivity::from(kuhn_tets()),
            &reference,
        )
            .into(),
        reference,
    )
        .into()
}

fn plastic_bar() -> Model<Plastic, 3> {
    let reference = coordinates().into();
    (
        (
            plasticity(),
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

mod plastic {
    use super::*;
    plastic_bar_tests!(plastic_bar, 1, 1.3, 5e-3);
    #[test]
    fn the_free_energy_follows_the_hyperelastic_one_until_it_yields()
    -> Result<(), crate::math::assert::AssertionError> {
        use crate::domain::solid::{
            elastic_plastic::ElasticPlasticElements as _,
            modal::test::{plastic_free_energy_follows_the_elastic, plastic_state},
        };
        let model = plastic_bar();
        plastic_free_energy_follows_the_elastic(
            &model,
            &model.initial_state(),
            &plastic_state::<1>(model.initial_state(), 1.3),
            &bar(),
        )
    }
}

mod viscoelastic {
    use super::*;
    viscoelastic_bar_tests!(viscoelastic_bar, 2, 5e-3);
}
