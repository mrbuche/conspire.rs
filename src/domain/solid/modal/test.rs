use super::{HyperelasticModes, HyperviscoelasticModes, ModalError};
use crate::{
    constitutive::{
        canonical::Canonical,
        fluid::{
            hyperviscous::Newtonian,
            plastic::{Linear, PlasticFlow, VonMises},
            viscoplastic::ViscoplasticFlow,
        },
        solid::{elastic::Elastic, hyperelastic::NeoHookean},
    },
    domain::{
        Model, NodalCoordinates,
        block::solid::viscoplastic::ViscoplasticStateVariables,
        solid::{
            NodalForcesSolid,
            elastic::ElasticElements,
            hyperelastic::HyperelasticElements,
            hyperelastic_plastic::HyperelasticPlasticElements,
            hyperviscoelastic::HyperviscoelasticElements,
            mass::{InertialMass, LumpedMassElements},
        },
    },
    math::{
        Current, Quantity, Tensor, TensorRank1Vec, Vector,
        assert::{Assert, AssertionError},
    },
    units::{Density, Rate, Stress, Viscosity},
};
use core::{array::from_fn, f64::consts::PI};

pub(crate) const ELEMENTS: usize = 40;
pub(crate) const LENGTH: f64 = 1.0;
pub(crate) const WIDTH: f64 = 0.1;
pub(crate) const SHEAR_MODULUS: f64 = 3.0e9;
pub(crate) const BULK_MODULUS: f64 = 13.0e9;
pub(crate) const DENSITY: f64 = 7.8e3;

pub(crate) fn wave_speed() -> f64 {
    ((BULK_MODULUS + 4.0 * SHEAR_MODULUS / 3.0) / DENSITY).sqrt()
}

pub(crate) fn exact(n: usize) -> f64 {
    (2 * n - 1) as f64 * PI * wave_speed() / (2.0 * LENGTH)
}

pub(crate) fn material() -> NeoHookean {
    NeoHookean {
        shear_modulus: Stress::pascals(SHEAR_MODULUS),
        bulk_modulus: Stress::pascals(BULK_MODULUS),
    }
}

pub(crate) fn density() -> Quantity<Density> {
    Density::kilograms_per_cubic_meter(DENSITY)
}

pub(crate) fn coordinates() -> Vec<[f64; 3]> {
    (0..=ELEMENTS)
        .flat_map(|k| {
            let z = LENGTH * k as f64 / ELEMENTS as f64;
            [
                [0.0, 0.0, z],
                [WIDTH, 0.0, z],
                [WIDTH, WIDTH, z],
                [0.0, WIDTH, z],
            ]
        })
        .collect()
}

pub(crate) fn kuhn_tets() -> Vec<[usize; 4]> {
    let nodes = coordinates();
    let volume = |[a, b, c, d]: [usize; 4]| {
        let edge = |i: usize| from_fn::<f64, 3, _>(|x| nodes[i][x] - nodes[a][x]);
        let (u, v, w) = (edge(b), edge(c), edge(d));
        u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0])
    };
    (0..ELEMENTS)
        .flat_map(|k| {
            let (a, b) = (4 * k, 4 * k + 4);
            [
                [a, a + 1, a + 2, b + 2],
                [a, a + 2, a + 3, b + 2],
                [a, a + 1, b + 1, b + 2],
                [a, a + 3, b + 3, b + 2],
                [a, b, b + 1, b + 2],
                [a, b, b + 3, b + 2],
            ]
        })
        .map(|[a, b, c, d]| {
            if volume([a, b, c, d]) > 0.0 {
                [a, b, c, d]
            } else {
                [a, c, b, d]
            }
        })
        .collect()
}

pub(crate) fn fixed() -> Vec<usize> {
    let nodes = 4 * (ELEMENTS + 1);
    (0..nodes)
        .flat_map(|node| [3 * node, 3 * node + 1])
        .chain((0..4).map(|node| 3 * node + 2))
        .collect()
}

pub(crate) fn at_rest() -> NodalCoordinates<3> {
    coordinates().into()
}

pub(crate) fn stretched(stretch: f64) -> NodalCoordinates<3> {
    coordinates()
        .into_iter()
        .map(|[x, y, z]| [x, y, stretch * z])
        .collect::<Vec<_>>()
        .into()
}

pub(crate) fn lowest() -> Quantity<Rate> {
    Quantity::new(0.0)
}

pub(crate) fn relatively(rel_tol: f64) -> Assert {
    Assert {
        abs_tol: 0.0,
        rel_tol,
        ..Assert::default()
    }
}

pub(crate) fn absolutely(abs_tol: f64) -> Assert {
    Assert {
        abs_tol,
        rel_tol: 0.0,
        ..Assert::default()
    }
}

pub(crate) fn tangent_modulus(stretch: f64) -> f64 {
    let gradient = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, stretch]].into();
    material()
        .first_piola_kirchhoff_tangent_stiffness(&gradient)
        .unwrap()[2][2][2][2]
        .value()
}

pub(crate) fn chain_of(modulus: f64, n: usize) -> f64 {
    let theta = (2 * n - 1) as f64 * PI / (2.0 * ELEMENTS as f64);
    2.0 * (modulus / DENSITY).sqrt() / (LENGTH / ELEMENTS as f64) * (theta / 2.0).sin()
}

pub(crate) fn chain(stretch: f64, n: usize) -> f64 {
    chain_of(tangent_modulus(stretch), n)
}

pub(crate) fn frequencies<B>(
    model: &Model<B, 3>,
    coordinates: &NodalCoordinates<3>,
    masses: &impl InertialMass<3>,
    near: Quantity<Rate>,
    count: usize,
) -> Result<Vec<f64>, AssertionError>
where
    B: HyperelasticElements<3>,
{
    Ok(model
        .modes(coordinates, masses, &fixed(), near, count)?
        .frequencies
        .iter()
        .map(|frequency| frequency.value())
        .collect())
}

fn flat(shape: &TensorRank1Vec<3, Current>) -> Vec<f64> {
    shape
        .iter()
        .flat_map(|node| node.iter().map(|entry| entry.value()))
        .collect()
}

fn stiffness_times<B>(model: &Model<B, 3>, coordinates: &NodalCoordinates<3>, x: &[f64]) -> Vec<f64>
where
    B: HyperelasticElements<3>,
{
    let tangent = model.nodal_stiffnesses(coordinates).unwrap();
    let mut product = vec![0.0; x.len()];
    tangent.iter().enumerate().for_each(|(a, row)| {
        row.entries().for_each(|(b, block)| {
            (0..3).for_each(|i| {
                (0..3).for_each(|j| product[3 * a + i] += block[i][j].value() * x[3 * b + j])
            })
        })
    });
    product
}

fn mass_times(masses: &impl InertialMass<3>, x: &[f64]) -> Vec<f64> {
    let forces: NodalForcesSolid<3> = masses.inertial_forces(&Vector::from(x).into());
    forces
        .iter()
        .flat_map(|node| node.iter().map(|entry| entry.value()))
        .collect()
}

pub(crate) fn check_shapes<B>(
    model: &Model<B, 3>,
    coordinates: &NodalCoordinates<3>,
    masses: impl InertialMass<3>,
) -> Result<(), AssertionError>
where
    B: HyperelasticElements<3>,
{
    let modes = model.modes(coordinates, &masses, &fixed(), lowest(), 3)?;
    let shapes: Vec<Vec<f64>> = modes.shapes.iter().map(flat).collect();
    let held = fixed();
    for (k, shape) in shapes.iter().enumerate() {
        held.iter()
            .try_for_each(|&index| Assert::zero(&shape[index]))?;
        let lambda = modes.frequencies[k].value().powi(2);
        let (kx, mx) = (
            stiffness_times(model, coordinates, shape),
            mass_times(&masses, shape),
        );
        let scale = kx.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let free = |values: Vec<f64>| -> Vector {
            values
                .into_iter()
                .enumerate()
                .filter(|(index, _)| !held.contains(index))
                .map(|(_, value)| value)
                .collect()
        };
        absolutely(1e-6 * scale)
            .eq_within_tols(free(kx), free(mx.iter().map(|mx| lambda * mx).collect()))?;
        for (l, other) in shapes.iter().enumerate() {
            let norm: f64 = mass_times(&masses, other)
                .iter()
                .zip(shape)
                .map(|(m, s)| m * s)
                .sum();
            let expected = if k == l { 1.0 } else { 0.0 };
            absolutely(1e-8).eq_within_tols(norm, expected)?;
        }
    }
    Ok(())
}

pub(crate) fn follows_the_chain<B>(
    model: &Model<B, 3>,
    masses: &impl InertialMass<3>,
    stretches: &[f64],
    highest: usize,
    tol: f64,
) -> Result<(), AssertionError>
where
    B: HyperelasticElements<3>,
{
    stretches.iter().try_for_each(|&stretch| {
        (1..=highest).try_for_each(|n| {
            let found = frequencies(
                model,
                &stretched(stretch),
                masses,
                Quantity::new(1.001 * chain(stretch, n)),
                1,
            )?;
            relatively(tol).eq_within_tols(found[0], chain(stretch, n))
        })
    })
}

pub(crate) fn is_below_the_continuous_bar<B>(
    model: &Model<B, 3>,
    masses: &impl InertialMass<3>,
) -> Result<(), AssertionError>
where
    B: HyperelasticElements<3>,
{
    (1..=3).try_for_each(|n| {
        let found = frequencies(model, &at_rest(), masses, Quantity::new(exact(n)), 1)?;
        assert!(found[0] < exact(n));
        Ok(())
    })
}

pub(crate) fn finds_the_modes_nearest_a_frequency<B>(
    model: &Model<B, 3>,
    masses: &impl InertialMass<3>,
) -> Result<(), AssertionError>
where
    B: HyperelasticElements<3>,
{
    let all = frequencies(model, &at_rest(), masses, lowest(), 12)?;
    let target = 0.5 * (all[4] + all[5]);
    let mut expected = all.clone();
    expected.sort_by(|a, b| (a - target).abs().total_cmp(&(b - target).abs()));
    let mut expected = expected[..3].to_vec();
    expected.sort_by(f64::total_cmp);
    let near = frequencies(model, &at_rest(), masses, Quantity::new(target), 3)?;
    relatively(1e-8).eq_within_tols(Vector::from(near), Vector::from(expected))
}

pub(crate) fn is_ascending<B>(
    model: &Model<B, 3>,
    masses: &impl InertialMass<3>,
) -> Result<(), AssertionError>
where
    B: HyperelasticElements<3>,
{
    let found = frequencies(model, &at_rest(), masses, lowest(), 5)?;
    assert!(found.windows(2).all(|pair| pair[0] < pair[1]));
    Ok(())
}

pub(crate) fn never_returns_a_wrong_mode_for_a_shift_on_a_frequency<B>(
    model: &Model<B, 3>,
    masses: &impl InertialMass<3>,
) -> Result<(), AssertionError>
where
    B: HyperelasticElements<3>,
{
    let all = frequencies(model, &at_rest(), masses, lowest(), 12)?;
    match model.modes(&at_rest(), masses, &fixed(), Quantity::new(all[4]), 3) {
        Err(ModalError::Unverified(_) | ModalError::Upstream(..)) => Ok(()),
        Err(error) => Err(error.into()),
        Ok(modes) => modes.frequencies.iter().try_for_each(|frequency| {
            let value = frequency.value();
            let nearest = all
                .iter()
                .copied()
                .min_by(|a, b| (a - value).abs().total_cmp(&(b - value).abs()))
                .unwrap_or(value);
            relatively(1e-8).eq_within_tols(value, nearest)
        }),
    }
}

pub(crate) fn stretching_moves_the_frequencies<B>(
    model: &Model<B, 3>,
    masses: &impl InertialMass<3>,
) -> Result<(), AssertionError>
where
    B: HyperelasticElements<3>,
{
    let rest = frequencies(model, &at_rest(), masses, lowest(), 1)?[0];
    let stretch = 1.02;
    let found = frequencies(
        model,
        &stretched(stretch),
        masses,
        Quantity::new(1.001 * chain(stretch, 1)),
        1,
    )?[0];
    assert!(relatively(5e-3).eq_within_tols(found, rest).is_err());
    Ok(())
}

#[test]
fn stretching_changes_the_tangent_modulus_enough_to_see() {
    [0.98, 1.02].iter().for_each(|&stretch| {
        assert!(
            relatively(5e-3)
                .eq_within_tols(chain(stretch, 1), chain(1.0, 1))
                .is_err()
        )
    });
}

pub(crate) fn flowing() -> Canonical<NeoHookean, ViscoplasticFlow> {
    (
        material(),
        ViscoplasticFlow {
            yield_stress: Stress::pascals(2.0),
            hardening_slope: Stress::pascals(1.0),
            rate_sensitivity: 0.25,
            reference_flow_rate: Rate::per_second(0.1),
        },
    )
        .into()
}

pub(crate) fn plasticity() -> Canonical<NeoHookean, PlasticFlow<VonMises, Linear>> {
    (
        material(),
        PlasticFlow {
            surface: VonMises,
            hardening: Linear {
                yield_stress: Stress::pascals(2.0),
                hardening_slope: Stress::pascals(1.0),
            },
        },
    )
        .into()
}

pub(crate) fn elastic_modulus(stretch: f64, plastic_stretch: f64) -> f64 {
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

pub(crate) fn plastic_state<const G: usize>(
    mut state: ViscoplasticStateVariables<G, Quantity>,
    plastic_stretch: f64,
) -> ViscoplasticStateVariables<G, Quantity> {
    let lateral = 1.0 / plastic_stretch.sqrt();
    state.iter_mut().for_each(|element| {
        element.iter_mut().for_each(|point| {
            point.0 = [
                [lateral, 0.0, 0.0],
                [0.0, lateral, 0.0],
                [0.0, 0.0, plastic_stretch],
            ]
            .into()
        })
    });
    state
}

pub(crate) fn plastic_free_energy_follows_the_elastic<P, E, S>(
    plastic: &Model<P, 3>,
    unyielded: &S,
    yielded: &S,
    elastic: &Model<E, 3>,
) -> Result<(), AssertionError>
where
    P: HyperelasticPlasticElements<S, 3>,
    E: HyperelasticElements<3>,
{
    let stretched = stretched(1.03);
    let scale = HyperelasticElements::helmholtz_free_energy(elastic, &stretched)?
        .value()
        .abs();
    [at_rest(), stretched.clone()]
        .iter()
        .try_for_each(|coordinates| {
            absolutely(1e-10 * scale).eq_within_tols(
                HyperelasticPlasticElements::helmholtz_free_energy(
                    plastic,
                    coordinates,
                    unyielded,
                )?
                .value(),
                HyperelasticElements::helmholtz_free_energy(elastic, coordinates)?.value(),
            )
        })?;
    let difference =
        HyperelasticPlasticElements::helmholtz_free_energy(plastic, &stretched, yielded)?.value()
            - HyperelasticElements::helmholtz_free_energy(elastic, &stretched)?.value();
    assert!(
        difference.abs() > 1e-3 * scale,
        "a plastic deformation did not change the free energy"
    );
    Ok(())
}

fn among(found: &[f64], expected: f64, tol: f64) -> Result<(), AssertionError> {
    let nearest = found
        .iter()
        .copied()
        .min_by(|a, b| (a - expected).abs().total_cmp(&(b - expected).abs()))
        .unwrap_or(expected);
    relatively(tol).eq_within_tols(nearest, expected)
}

pub(crate) fn unyielded_follows_the_chain(
    find: &impl Fn(f64, f64, usize) -> Result<Vec<f64>, AssertionError>,
    tol: f64,
) -> Result<(), AssertionError> {
    [1.0, 0.98, 1.02].iter().try_for_each(|&stretch| {
        (1..=3).try_for_each(|n| {
            among(
                &find(stretch, 1.001 * chain(stretch, n), 4)?,
                chain(stretch, n),
                tol,
            )
        })
    })
}

pub(crate) fn plastic_follows_the_chain(
    find: &impl Fn(f64, f64, usize) -> Result<Vec<f64>, AssertionError>,
    plastic_stretch: f64,
    tol: f64,
) -> Result<(), AssertionError> {
    [1.0, 1.03].iter().try_for_each(|&stretch| {
        (1..=3).try_for_each(|n| {
            let expected = chain_of(elastic_modulus(stretch, plastic_stretch), n);
            among(&find(stretch, 1.001 * expected, 4)?, expected, tol)
        })
    })
}

pub(crate) fn plastic_state_moves_the_frequencies(
    find: &impl Fn(f64, f64, usize) -> Result<Vec<f64>, AssertionError>,
    plastic_stretch: f64,
    tol: f64,
) -> Result<(), AssertionError> {
    let unyielded = chain(1.0, 1);
    let yielded = chain_of(elastic_modulus(1.0, plastic_stretch), 1);
    assert!(
        relatively(2.0 * tol.max(1.5e-3))
            .eq_within_tols(yielded, unyielded)
            .is_err()
    );
    among(&find(1.0, 1.001 * yielded, 1)?, yielded, tol)
}

pub(crate) const BULK_VISCOSITY: f64 = 5.0e4;
pub(crate) const SHEAR_VISCOSITY: f64 = 1.0e5;

pub(crate) fn viscoelasticity() -> Canonical<NeoHookean, Newtonian> {
    Canonical::from((
        material(),
        Newtonian {
            bulk_viscosity: Viscosity::pascal_seconds(BULK_VISCOSITY),
            shear_viscosity: Viscosity::pascal_seconds(SHEAR_VISCOSITY),
        },
    ))
}

pub(crate) fn viscoelastic_frequencies_follow_the_chain<B>(
    model: &Model<B, 3>,
    tol: f64,
) -> Result<(), AssertionError>
where
    B: HyperviscoelasticElements<3> + LumpedMassElements<3>,
{
    (1..=4).try_for_each(|n| {
        let expected = chain(1.0, n);
        let found = model
            .modes(
                &at_rest(),
                &model.nodal_lumped_masses(),
                &fixed(),
                Quantity::new(1.001 * expected),
                4,
            )?
            .modes
            .frequencies
            .iter()
            .map(|frequency| frequency.value())
            .collect::<Vec<_>>();
        among(&found, expected, tol)
    })
}

pub(crate) fn viscoelastic_damping_follows_the_viscosity<B>(
    model: &Model<B, 3>,
    highest: usize,
    tol: f64,
) -> Result<(), AssertionError>
where
    B: HyperviscoelasticElements<3> + LumpedMassElements<3>,
{
    let constrained = BULK_MODULUS + 4.0 * SHEAR_MODULUS / 3.0;
    let viscous = BULK_VISCOSITY + 4.0 * SHEAR_VISCOSITY / 3.0;
    (1..=highest).try_for_each(|n| {
        let frequency = chain(1.0, n);
        let modes = model.modes(
            &at_rest(),
            &model.nodal_lumped_masses(),
            &fixed(),
            Quantity::new(1.001 * frequency),
            4,
        )?;
        let ratio = viscous * frequency / (2.0 * constrained);
        let damped = modes.damped_frequencies();
        let found = modes
            .modes
            .frequencies
            .iter()
            .zip(&modes.damping_ratios)
            .zip(&damped)
            .any(|((undamped, &damping), damped)| {
                relatively(tol)
                    .eq_within_tols(undamped.value(), frequency)
                    .is_ok()
                    && relatively(tol).eq_within_tols(damping, ratio).is_ok()
                    && relatively(tol)
                        .eq_within_tols(damped.value(), frequency * (1.0 - ratio * ratio).sqrt())
                        .is_ok()
            });
        if found {
            Ok(())
        } else {
            Err(format!(
                "No mode near {frequency} has the damping ratio {ratio}: {:?}",
                modes.damping_ratios
            )
            .into())
        }
    })
}

macro_rules! viscoelastic_bar_tests {
    ($model:path, $highest:expr, $tol:expr) => {
        #[test]
        fn the_undamped_frequencies_follow_the_chain_of_the_bar()
        -> Result<(), $crate::math::assert::AssertionError> {
            $crate::domain::solid::modal::test::viscoelastic_frequencies_follow_the_chain(
                &$model(),
                $tol,
            )
        }
        #[test]
        fn the_damping_ratios_follow_the_viscosity_of_the_bar()
        -> Result<(), $crate::math::assert::AssertionError> {
            $crate::domain::solid::modal::test::viscoelastic_damping_follows_the_viscosity(
                &$model(),
                $highest,
                $tol,
            )
        }
    };
}

pub(crate) use viscoelastic_bar_tests;

macro_rules! inelastic_bar_tests {
    ($model:path, $points:literal, $plastic:expr, $tol:expr, $elements:path, $modes:path) => {
        #[test]
        fn an_unyielded_bar_follows_the_chain_of_its_elastic_model()
        -> Result<(), $crate::math::assert::AssertionError> {
            use $elements as _;
            use $modes as _;
            let model = $model();
            let state = model.initial_state();
            let find = |stretch: f64, near: f64, count: usize| {
                Ok(model
                    .modes(
                        &$crate::domain::solid::modal::test::stretched(stretch),
                        &state,
                        &model.nodal_lumped_masses(),
                        &$crate::domain::solid::modal::test::fixed(),
                        $crate::math::Quantity::new(near),
                        count,
                    )?
                    .frequencies
                    .iter()
                    .map(|frequency| frequency.value())
                    .collect())
            };
            $crate::domain::solid::modal::test::unyielded_follows_the_chain(&find, $tol)
        }
        #[test]
        fn a_plastic_state_follows_the_chain_of_its_elastic_tangent()
        -> Result<(), $crate::math::assert::AssertionError> {
            use $elements as _;
            use $modes as _;
            let model = $model();
            let state = $crate::domain::solid::modal::test::plastic_state::<$points>(
                model.initial_state(),
                $plastic,
            );
            let find = |stretch: f64, near: f64, count: usize| {
                Ok(model
                    .modes(
                        &$crate::domain::solid::modal::test::stretched(stretch),
                        &state,
                        &model.nodal_lumped_masses(),
                        &$crate::domain::solid::modal::test::fixed(),
                        $crate::math::Quantity::new(near),
                        count,
                    )?
                    .frequencies
                    .iter()
                    .map(|frequency| frequency.value())
                    .collect())
            };
            $crate::domain::solid::modal::test::plastic_follows_the_chain(&find, $plastic, $tol)
        }
        #[test]
        fn a_plastic_state_moves_the_frequencies()
        -> Result<(), $crate::math::assert::AssertionError> {
            use $elements as _;
            use $modes as _;
            let model = $model();
            let state = $crate::domain::solid::modal::test::plastic_state::<$points>(
                model.initial_state(),
                $plastic,
            );
            let find = |stretch: f64, near: f64, count: usize| {
                Ok(model
                    .modes(
                        &$crate::domain::solid::modal::test::stretched(stretch),
                        &state,
                        &model.nodal_lumped_masses(),
                        &$crate::domain::solid::modal::test::fixed(),
                        $crate::math::Quantity::new(near),
                        count,
                    )?
                    .frequencies
                    .iter()
                    .map(|frequency| frequency.value())
                    .collect())
            };
            $crate::domain::solid::modal::test::plastic_state_moves_the_frequencies(
                &find, $plastic, $tol,
            )
        }
    };
}

macro_rules! viscoplastic_bar_tests {
    ($model:path, $points:literal, $plastic:expr, $tol:expr) => {
        $crate::domain::solid::modal::test::inelastic_bar_tests!(
            $model,
            $points,
            $plastic,
            $tol,
            $crate::domain::solid::elastic_viscoplastic::ElasticViscoplasticElements,
            $crate::domain::solid::modal::HyperelasticViscoplasticModes
        );
    };
}

macro_rules! plastic_bar_tests {
    ($model:path, $points:literal, $plastic:expr, $tol:expr) => {
        $crate::domain::solid::modal::test::inelastic_bar_tests!(
            $model,
            $points,
            $plastic,
            $tol,
            $crate::domain::solid::elastic_plastic::ElasticPlasticElements,
            $crate::domain::solid::modal::HyperelasticPlasticModes
        );
    };
}

pub(crate) use {inelastic_bar_tests, plastic_bar_tests, viscoplastic_bar_tests};

macro_rules! lumped_bar_tests {
    ($model:path, $tol:expr) => {
        #[test]
        fn the_lumped_mass_follows_the_chain_of_the_bar()
        -> Result<(), $crate::math::assert::AssertionError> {
            let model = $model();
            $crate::domain::solid::modal::test::follows_the_chain(
                &model,
                &model.nodal_lumped_masses(),
                &[1.0],
                4,
                $tol,
            )
        }
        #[test]
        fn modes_nearest_a_frequency_are_found_around_it()
        -> Result<(), $crate::math::assert::AssertionError> {
            let model = $model();
            $crate::domain::solid::modal::test::finds_the_modes_nearest_a_frequency(
                &model,
                &model.nodal_lumped_masses(),
            )
        }
        #[test]
        fn a_shift_on_a_natural_frequency_never_gives_a_wrong_mode()
        -> Result<(), $crate::math::assert::AssertionError> {
            let model = $model();
            $crate::domain::solid::modal::test::never_returns_a_wrong_mode_for_a_shift_on_a_frequency(
                &model,
                &model.nodal_lumped_masses(),
            )
        }
        #[test]
        fn the_modes_are_ascending() -> Result<(), $crate::math::assert::AssertionError> {
            let model = $model();
            $crate::domain::solid::modal::test::is_ascending(&model, &model.nodal_lumped_masses())
        }
        #[test]
        fn shapes_solve_the_eigenproblem_and_are_mass_orthonormal()
        -> Result<(), $crate::math::assert::AssertionError> {
            let model = $model();
            $crate::domain::solid::modal::test::check_shapes(
                &model,
                &$crate::domain::solid::modal::test::at_rest(),
                model.nodal_lumped_masses(),
            )
        }
        #[test]
        fn a_stretched_bar_follows_the_chain_of_its_tangent_modulus()
        -> Result<(), $crate::math::assert::AssertionError> {
            let model = $model();
            $crate::domain::solid::modal::test::follows_the_chain(
                &model,
                &model.nodal_lumped_masses(),
                &[0.98, 1.02, 1.05],
                3,
                $tol,
            )
        }
        #[test]
        fn a_stretched_bar_is_not_at_its_unstressed_frequencies()
        -> Result<(), $crate::math::assert::AssertionError> {
            let model = $model();
            $crate::domain::solid::modal::test::stretching_moves_the_frequencies(
                &model,
                &model.nodal_lumped_masses(),
            )
        }
        #[test]
        fn shapes_solve_the_eigenproblem_about_the_stretched_state()
        -> Result<(), $crate::math::assert::AssertionError> {
            let model = $model();
            $crate::domain::solid::modal::test::check_shapes(
                &model,
                &$crate::domain::solid::modal::test::stretched(1.02),
                model.nodal_lumped_masses(),
            )
        }
    };
    ($model:path, $tol:expr, below) => {
        $crate::domain::solid::modal::test::lumped_bar_tests!($model, $tol);
        #[test]
        fn the_lumped_mass_is_below_the_continuous_bar()
        -> Result<(), $crate::math::assert::AssertionError> {
            let model = $model();
            $crate::domain::solid::modal::test::is_below_the_continuous_bar(
                &model,
                &model.nodal_lumped_masses(),
            )
        }
    };
}

pub(crate) use lumped_bar_tests;
