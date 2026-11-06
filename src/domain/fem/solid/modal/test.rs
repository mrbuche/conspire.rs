use crate::{
    domain::solid::modal::test::{
        ELEMENTS, at_rest, check_shapes, coordinates, density, exact, flowing, frequencies,
        kuhn_tets, lumped_bar_tests, material, plastic_bar_tests, plasticity, relatively,
        viscoelastic_bar_tests, viscoelasticity, viscoplastic_bar_tests,
    },
    fem::{
        Model,
        block::{
            Block,
            element::linear::{Hexahedron, Tetrahedron},
        },
    },
    math::{Quantity, assert::AssertionError},
    units::Density,
};
use core::array::from_fn;

type Hexes = Block<
    crate::constitutive::solid::hyperelastic::NeoHookean,
    Hexahedron,
    8,
    3,
    8,
    8,
    Quantity<Density>,
>;
type Tets = Block<
    crate::constitutive::solid::hyperelastic::NeoHookean,
    Tetrahedron<4>,
    4,
    3,
    4,
    4,
    Quantity<Density>,
>;

fn hexes() -> Model<Hexes, 3> {
    let reference = coordinates().into();
    (
        (
            material(),
            density(),
            (0..ELEMENTS)
                .map(|k| from_fn(|i| 4 * k + i))
                .collect::<Vec<[usize; 8]>>(),
            &reference,
        )
            .into(),
        reference,
    )
        .into()
}

fn tets() -> Model<Tets, 3> {
    let reference = coordinates().into();
    (
        Tets::from((material(), density(), kuhn_tets(), &reference)),
        reference,
    )
        .into()
}

mod hexahedra {
    use super::*;
    lumped_bar_tests!(hexes, 1e-8, below);
}

mod tetrahedra {
    use super::*;
    #[test]
    fn the_consistent_mass_is_close_to_the_continuous_bar() -> Result<(), AssertionError> {
        let model = tets();
        (1..=4).try_for_each(|n| {
            let found = frequencies(
                &model,
                &at_rest(),
                &model.nodal_masses(),
                Quantity::new(exact(n)),
                1,
            )?;
            relatively(3e-3).eq_within_tols(found[0], exact(n))
        })
    }
    #[test]
    fn the_lumped_mass_is_below_and_the_consistent_mass_above_the_continuous_bar()
    -> Result<(), AssertionError> {
        let model = tets();
        (1..=3).try_for_each(|n| {
            let near = Quantity::new(exact(n));
            let lumped = frequencies(&model, &at_rest(), &model.nodal_lumped_masses(), near, 1)?;
            let consistent = frequencies(&model, &at_rest(), &model.nodal_masses(), near, 1)?;
            assert!(lumped[0] < exact(n));
            assert!(exact(n) < consistent[0]);
            Ok(())
        })
    }
    #[test]
    fn shapes_solve_the_eigenproblem_and_are_mass_orthonormal() -> Result<(), AssertionError> {
        let model = tets();
        check_shapes(&model, &at_rest(), model.nodal_masses())
    }
}

mod viscoplastic {
    use super::*;
    use crate::constitutive::{
        canonical::Canonical, fluid::viscoplastic::ViscoplasticFlow,
        solid::hyperelastic::NeoHookean,
    };
    type Bar =
        Block<Canonical<NeoHookean, ViscoplasticFlow>, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
    fn bar() -> Model<Bar, 3> {
        let reference = coordinates().into();
        (
            Bar::from((
                flowing(),
                density(),
                (0..ELEMENTS)
                    .map(|k| from_fn(|i| 4 * k + i))
                    .collect::<Vec<[usize; 8]>>(),
                &reference,
            )),
            reference,
        )
            .into()
    }
    viscoplastic_bar_tests!(bar, 8, 1.1, 1e-8);
}

mod plastic {
    use super::*;
    use crate::constitutive::{
        canonical::Canonical,
        fluid::plastic::{Linear, PlasticFlow, VonMises},
        solid::hyperelastic::NeoHookean,
    };
    type Bar = Block<
        Canonical<NeoHookean, PlasticFlow<VonMises, Linear>>,
        Hexahedron,
        8,
        3,
        8,
        8,
        Quantity<Density>,
    >;
    fn bar() -> Model<Bar, 3> {
        let reference = coordinates().into();
        (
            (
                plasticity(),
                density(),
                (0..ELEMENTS)
                    .map(|k| from_fn(|i| 4 * k + i))
                    .collect::<Vec<[usize; 8]>>(),
                &reference,
            )
                .into(),
            reference,
        )
            .into()
    }
    plastic_bar_tests!(bar, 8, 1.1, 1e-8);
    #[test]
    fn the_free_energy_follows_the_hyperelastic_one_until_it_yields() -> Result<(), AssertionError>
    {
        use crate::domain::solid::{
            elastic_plastic::ElasticPlasticElements as _,
            modal::test::{plastic_free_energy_follows_the_elastic, plastic_state},
        };
        let model = bar();
        plastic_free_energy_follows_the_elastic(
            &model,
            &model.initial_state(),
            &plastic_state::<8>(model.initial_state(), 1.1),
            &hexes(),
        )
    }
}

mod viscoelastic {
    use super::*;
    use crate::constitutive::{
        canonical::Canonical, fluid::hyperviscous::Newtonian, solid::hyperelastic::NeoHookean,
    };
    type Bar = Block<Canonical<NeoHookean, Newtonian>, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
    fn bar() -> Model<Bar, 3> {
        let reference = coordinates().into();
        (
            (
                viscoelasticity(),
                density(),
                (0..ELEMENTS)
                    .map(|k| from_fn(|i| 4 * k + i))
                    .collect::<Vec<[usize; 8]>>(),
                &reference,
            )
                .into(),
            reference,
        )
            .into()
    }
    viscoelastic_bar_tests!(bar, 3, 1e-8);
}
