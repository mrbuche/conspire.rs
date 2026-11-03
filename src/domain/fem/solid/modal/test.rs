use super::ElasticModes;
use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    domain::solid::{NodalForcesSolid, mass::InertialMass},
    fem::{
        Model, NodalCoordinates, NodalReferenceCoordinates,
        block::{
            Block,
            element::linear::{Hexahedron, Tetrahedron},
        },
        solid::elastic::ElasticElements,
    },
    math::{
        Current, Quantity, Tensor, TensorRank1Vec, Vector,
        assert::{Assert, AssertionError},
    },
    units::{Density, Rate, Stress},
};
use std::{array::from_fn, f64::consts::PI};

type Hexes = Block<NeoHookean, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type Tets = Block<NeoHookean, Tetrahedron<4>, 4, 3, 4, 4, Quantity<Density>>;

const ELEMENTS: usize = 40;
const LENGTH: f64 = 1.0;
const WIDTH: f64 = 0.1;
const SHEAR_MODULUS: f64 = 3.0e9;
const BULK_MODULUS: f64 = 13.0e9;
const DENSITY: f64 = 7.8e3;

fn wave_speed() -> f64 {
    ((BULK_MODULUS + 4.0 * SHEAR_MODULUS / 3.0) / DENSITY).sqrt()
}

fn exact(n: usize) -> f64 {
    (2 * n - 1) as f64 * PI * wave_speed() / (2.0 * LENGTH)
}

fn coordinates() -> Vec<[f64; 3]> {
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

fn material() -> NeoHookean {
    NeoHookean {
        shear_modulus: Stress::pascals(SHEAR_MODULUS),
        bulk_modulus: Stress::pascals(BULK_MODULUS),
    }
}

fn density() -> Quantity<Density> {
    Density::kilograms_per_cubic_meter(DENSITY)
}

fn hexes() -> Model<Hexes, 3> {
    let reference = NodalReferenceCoordinates::from(coordinates());
    (
        Hexes::from((
            material(),
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

fn tets() -> Model<Tets, 3> {
    let nodes = coordinates();
    let volume = |[a, b, c, d]: [usize; 4]| {
        let edge = |i: usize| from_fn::<f64, 3, _>(|x| nodes[i][x] - nodes[a][x]);
        let (u, v, w) = (edge(b), edge(c), edge(d));
        u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0])
    };
    let connectivity: Vec<[usize; 4]> = (0..ELEMENTS)
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
        .collect();
    let reference = NodalReferenceCoordinates::from(nodes);
    (
        Tets::from((material(), density(), connectivity, &reference)),
        reference,
    )
        .into()
}

fn fixed() -> Vec<usize> {
    let nodes = 4 * (ELEMENTS + 1);
    (0..nodes)
        .flat_map(|node| [3 * node, 3 * node + 1])
        .chain((0..4).map(|node| 3 * node + 2))
        .collect()
}

fn at_rest() -> NodalCoordinates<3> {
    coordinates().into()
}

fn stretched(stretch: f64) -> NodalCoordinates<3> {
    coordinates()
        .into_iter()
        .map(|[x, y, z]| [x, y, stretch * z])
        .collect::<Vec<_>>()
        .into()
}

fn lowest() -> Quantity<Rate> {
    Quantity::new(0.0)
}

fn frequencies<B>(
    model: &Model<B, 3>,
    masses: &impl InertialMass<3>,
    near: Quantity<Rate>,
    count: usize,
) -> Vec<f64>
where
    B: ElasticElements<3>,
{
    model
        .modes(&at_rest(), masses, &fixed(), near, count)
        .unwrap()
        .frequencies
        .iter()
        .map(|frequency| frequency.value())
        .collect()
}

fn relatively(rel_tol: f64) -> Assert {
    Assert {
        abs_tol: 0.0,
        rel_tol,
        ..Assert::default()
    }
}

fn absolutely(abs_tol: f64) -> Assert {
    Assert {
        abs_tol,
        rel_tol: 0.0,
        ..Assert::default()
    }
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

fn check_shapes<B>(
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
        absolutely(1e-8 * scale)
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

mod hexahedra {
    use super::*;

    #[test]
    fn the_lumped_mass_follows_the_discrete_chain() -> Result<(), AssertionError> {
        let model = hexes();
        let theta = |n: usize| (2 * n - 1) as f64 * PI / (2.0 * ELEMENTS as f64);
        let h = LENGTH / ELEMENTS as f64;
        (1..=4).try_for_each(|n| {
            let chain = 2.0 * wave_speed() / h * (theta(n) / 2.0).sin();
            let found = frequencies(
                &model,
                &model.nodal_lumped_masses(),
                Quantity::new(1.001 * chain),
                1,
            );
            relatively(1e-8).eq_within_tols(found[0], chain)
        })
    }

    #[test]
    fn the_lumped_mass_is_below_the_continuous_bar() {
        let model = hexes();
        (1..=3).for_each(|n| {
            let found = frequencies(
                &model,
                &model.nodal_lumped_masses(),
                Quantity::new(exact(n)),
                1,
            );
            assert!(found[0] < exact(n));
        });
    }

    #[test]
    fn modes_nearest_a_frequency_are_found_around_it() -> Result<(), AssertionError> {
        let model = hexes();
        let masses = model.nodal_lumped_masses();
        let all = frequencies(&model, &masses, lowest(), 12);
        let target = 0.5 * (all[4] + all[5]);
        let mut expected = all.clone();
        expected.sort_by(|a, b| (a - target).abs().total_cmp(&(b - target).abs()));
        let mut expected = expected[..3].to_vec();
        expected.sort_by(f64::total_cmp);
        let near = frequencies(&model, &masses, Quantity::new(target), 3);
        relatively(1e-8).eq_within_tols(Vector::from(near), Vector::from(expected))
    }

    #[test]
    fn the_modes_are_ascending() {
        let model = hexes();
        let frequencies = frequencies(&model, &model.nodal_lumped_masses(), lowest(), 5);
        assert!(frequencies.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn shapes_solve_the_eigenproblem_and_are_mass_orthonormal() -> Result<(), AssertionError> {
        let model = hexes();
        check_shapes(&model, &at_rest(), model.nodal_lumped_masses())
    }
}

mod tetrahedra {
    use super::*;

    #[test]
    fn the_consistent_mass_is_close_to_the_continuous_bar() -> Result<(), AssertionError> {
        let model = tets();
        (1..=4).try_for_each(|n| {
            let found = frequencies(&model, &model.nodal_masses(), Quantity::new(exact(n)), 1);
            relatively(3e-3).eq_within_tols(found[0], exact(n))
        })
    }

    #[test]
    fn the_lumped_mass_is_below_and_the_consistent_mass_above_the_continuous_bar() {
        let model = tets();
        for n in 1..=3 {
            let near = Quantity::new(exact(n));
            let lumped = frequencies(&model, &model.nodal_lumped_masses(), near, 1);
            let consistent = frequencies(&model, &model.nodal_masses(), near, 1);
            assert!(lumped[0] < exact(n));
            assert!(exact(n) < consistent[0]);
        }
    }

    #[test]
    fn shapes_solve_the_eigenproblem_and_are_mass_orthonormal() -> Result<(), AssertionError> {
        let model = tets();
        check_shapes(&model, &at_rest(), model.nodal_masses())
    }
}

mod prestressed {
    use super::*;
    use crate::constitutive::solid::elastic::Elastic;
    fn tangent_modulus(stretch: f64) -> f64 {
        let gradient = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, stretch]].into();
        material()
            .first_piola_kirchhoff_tangent_stiffness(&gradient)
            .unwrap()[2][2][2][2]
            .value()
    }
    fn chain(stretch: f64, n: usize) -> f64 {
        let theta = (2 * n - 1) as f64 * PI / (2.0 * ELEMENTS as f64);
        2.0 * (tangent_modulus(stretch) / DENSITY).sqrt() / (LENGTH / ELEMENTS as f64)
            * (theta / 2.0).sin()
    }
    fn found(stretch: f64, n: usize) -> f64 {
        let model = hexes();
        model
            .modes(
                &stretched(stretch),
                &model.nodal_lumped_masses(),
                &fixed(),
                Quantity::new(1.001 * chain(stretch, n)),
                1,
            )
            .unwrap()
            .frequencies[0]
            .value()
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
    #[test]
    fn a_stretched_bar_follows_the_chain_of_its_tangent_modulus() -> Result<(), AssertionError> {
        [0.98, 1.02, 1.05].iter().try_for_each(|&stretch| {
            (1..=3).try_for_each(|n| {
                relatively(1e-8).eq_within_tols(found(stretch, n), chain(stretch, n))
            })
        })
    }
    #[test]
    fn a_stretched_bar_is_not_at_its_unstressed_frequencies() -> Result<(), AssertionError> {
        let model = hexes();
        let masses = model.nodal_lumped_masses();
        let rest = model
            .modes(&at_rest(), &masses, &fixed(), lowest(), 1)?
            .frequencies[0]
            .value();
        assert!(
            relatively(5e-3)
                .eq_within_tols(found(1.02, 1), rest)
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn shapes_solve_the_eigenproblem_about_the_stretched_state() -> Result<(), AssertionError> {
        let model = hexes();
        check_shapes(&model, &stretched(1.02), model.nodal_lumped_masses())
    }
}
