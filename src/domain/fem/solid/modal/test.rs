use crate::{
    domain::solid::modal::test::{
        ELEMENTS, at_rest, check_shapes, coordinates, density, exact, frequencies, kuhn_tets,
        lumped_bar_tests, material, relatively,
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
use std::array::from_fn;

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
    use crate::{
        constitutive::{
            canonical::Canonical,
            fluid::viscoplastic::ViscoplasticFlow,
            solid::{elastic::Elastic, hyperelastic::NeoHookean},
        },
        domain::solid::modal::test::{chain, chain_of, fixed, stretched},
        fem::solid::{
            elastic_viscoplastic::ElasticViscoplasticElements, modal::ElasticViscoplasticModes,
        },
        units::{Rate, Stress},
    };
    use crate::{domain::block::solid::viscoplastic::ViscoplasticStateVariables, math::Tensor};
    type Flowing = Canonical<NeoHookean, ViscoplasticFlow>;
    type Bar = Block<Flowing, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
    fn bar() -> Model<Bar, 3> {
        let reference = coordinates().into();
        (
            Bar::from((
                Flowing::from((
                    material(),
                    ViscoplasticFlow {
                        yield_stress: Stress::pascals(2.0),
                        hardening_slope: Stress::pascals(1.0),
                        rate_sensitivity: 0.25,
                        reference_flow_rate: Rate::per_second(0.1),
                    },
                )),
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
    fn plastic(model: &Model<Bar, 3>, stretch: f64) -> ViscoplasticStateVariables<8, Quantity> {
        let lateral = 1.0 / stretch.sqrt();
        let mut state = model.initial_state();
        state.iter_mut().for_each(|element| {
            element.iter_mut().for_each(|point| {
                point.0 = [
                    [lateral, 0.0, 0.0],
                    [0.0, lateral, 0.0],
                    [0.0, 0.0, stretch],
                ]
                .into()
            })
        });
        state
    }
    fn found(
        model: &Model<Bar, 3>,
        stretch: f64,
        state: &ViscoplasticStateVariables<8, Quantity>,
        near: f64,
        count: usize,
    ) -> Result<Vec<f64>, AssertionError> {
        Ok(model
            .modes(
                &stretched(stretch),
                state,
                &model.nodal_lumped_masses(),
                &fixed(),
                Quantity::new(near),
                count,
            )?
            .frequencies
            .iter()
            .map(|frequency| frequency.value())
            .collect())
    }
    fn among(found: &[f64], expected: f64) -> Result<(), AssertionError> {
        let nearest = found
            .iter()
            .copied()
            .min_by(|a, b| (a - expected).abs().total_cmp(&(b - expected).abs()))
            .unwrap_or(expected);
        relatively(1e-8).eq_within_tols(nearest, expected)
    }
    fn elastic_modulus(stretch: f64, plastic_stretch: f64) -> f64 {
        let lateral = plastic_stretch.sqrt();
        let elastic = [
            [lateral, 0.0, 0.0],
            [0.0, lateral, 0.0],
            [0.0, 0.0, stretch / plastic_stretch],
        ]
        .into();
        material()
            .first_piola_kirchhoff_tangent_stiffness(&elastic)
            .unwrap()[2][2][2][2]
            .value()
            / plastic_stretch.powi(2)
    }
    #[test]
    fn an_unyielded_bar_follows_the_chain_of_its_elastic_model() -> Result<(), AssertionError> {
        let model = bar();
        let state = model.initial_state();
        [1.0, 0.98, 1.02].iter().try_for_each(|&stretch| {
            (1..=3).try_for_each(|n| {
                among(
                    &found(&model, stretch, &state, 1.001 * chain(stretch, n), 4)?,
                    chain(stretch, n),
                )
            })
        })
    }
    #[test]
    fn a_plastic_state_follows_the_chain_of_its_elastic_tangent() -> Result<(), AssertionError> {
        let model = bar();
        let plastic_stretch = 1.1;
        let state = plastic(&model, plastic_stretch);
        [1.0, 1.03].iter().try_for_each(|&stretch| {
            (1..=3).try_for_each(|n| {
                let expected = chain_of(elastic_modulus(stretch, plastic_stretch), n);
                among(
                    &found(&model, stretch, &state, 1.001 * expected, 4)?,
                    expected,
                )
            })
        })
    }
    #[test]
    fn a_plastic_state_moves_the_frequencies() -> Result<(), AssertionError> {
        let model = bar();
        let unyielded = chain(1.0, 1);
        let yielded = chain_of(elastic_modulus(1.0, 1.1), 1);
        assert!(relatively(3e-3).eq_within_tols(yielded, unyielded).is_err());
        let state = plastic(&model, 1.1);
        among(&found(&model, 1.0, &state, 1.001 * yielded, 1)?, yielded)
    }
}
