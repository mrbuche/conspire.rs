use super::ElasticModes;
use crate::{
    constitutive::solid::{elastic::Elastic, hyperelastic::NeoHookean},
    domain::{
        Model, NodalCoordinates,
        solid::{NodalForcesSolid, elastic::ElasticElements, mass::InertialMass},
    },
    math::{
        Current, Quantity, Tensor, TensorRank1Vec, Vector,
        assert::{Assert, AssertionError},
    },
    units::{Density, Rate, Stress},
};
use std::{array::from_fn, f64::consts::PI};

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

pub(crate) fn chain(stretch: f64, n: usize) -> f64 {
    let theta = (2 * n - 1) as f64 * PI / (2.0 * ELEMENTS as f64);
    2.0 * (tangent_modulus(stretch) / DENSITY).sqrt() / (LENGTH / ELEMENTS as f64)
        * (theta / 2.0).sin()
}

pub(crate) fn frequencies<B>(
    model: &Model<B, 3>,
    coordinates: &NodalCoordinates<3>,
    masses: &impl InertialMass<3>,
    near: Quantity<Rate>,
    count: usize,
) -> Result<Vec<f64>, AssertionError>
where
    B: ElasticElements<3>,
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
    B: ElasticElements<3>,
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
    B: ElasticElements<3>,
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
    B: ElasticElements<3>,
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
    B: ElasticElements<3>,
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
    B: ElasticElements<3>,
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
    B: ElasticElements<3>,
{
    let found = frequencies(model, &at_rest(), masses, lowest(), 5)?;
    assert!(found.windows(2).all(|pair| pair[0] < pair[1]));
    Ok(())
}

pub(crate) fn stretching_moves_the_frequencies<B>(
    model: &Model<B, 3>,
    masses: &impl InertialMass<3>,
) -> Result<(), AssertionError>
where
    B: ElasticElements<3>,
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
