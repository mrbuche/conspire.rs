use super::{fastest_time_scale, largest_eigenvalue};
use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalReferenceCoordinates,
        linear::{Hexahedron, Tetrahedron},
        mass::{ElementNodalLumpedMasses, IntegrationDensities, LumpedMassFiniteElement},
        solid::{ElementNodalStiffnessesSolid, elastic::ElasticElement},
    },
    math::Quantity,
    units::{Density, Stress},
};

const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);

fn material() -> NeoHookean {
    NeoHookean {
        shear_modulus: Stress::pascals(3.0e9),
        bulk_modulus: Stress::pascals(13.0e9),
    }
}

fn hexahedron_reference() -> [[f64; 3]; 8] {
    [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 3.0],
        [2.0, 0.0, 3.0],
        [2.0, 1.0, 3.0],
        [0.0, 1.0, 3.0],
    ]
}

fn tetrahedron_reference() -> [[f64; 3]; 4] {
    [
        [0.1, 0.2, 0.0],
        [1.3, 0.1, 0.2],
        [0.2, 0.9, 0.1],
        [0.3, 0.4, 1.2],
    ]
}

fn deform<const N: usize>(reference: [[f64; 3]; N]) -> [[f64; 3]; N] {
    reference.map(|[x, y, z]| {
        [
            1.02 * x + 0.03 * y - 0.01 * z + 0.01 * y * z,
            0.98 * y - 0.02 * x + 0.02 * z,
            1.01 * z + 0.02 * x - 0.01 * x * y,
        ]
    })
}

fn scale<const N: usize>(coordinates: [[f64; 3]; N], factor: f64) -> [[f64; 3]; N] {
    coordinates.map(|coordinate| coordinate.map(|component| factor * component))
}

fn jacobi(stiffnesses: &[Vec<f64>], masses: &[f64]) -> f64 {
    let n = masses.len();
    let mut a: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    0.5 * (stiffnesses[i][j] + stiffnesses[j][i]) / (masses[i] * masses[j]).sqrt()
                })
                .collect()
        })
        .collect();
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        let diagonal: f64 = (0..n).map(|i| a[i][i] * a[i][i]).sum();
        if off <= 1e-26 * diagonal {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                a.iter_mut().for_each(|row| {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                });
                let (head, tail) = a.split_at_mut(q);
                head[p]
                    .iter_mut()
                    .zip(tail[0].iter_mut())
                    .for_each(|(apk, aqk)| {
                        let (x, y) = (*apk, *aqk);
                        *apk = c * x - s * y;
                        *aqk = s * x + c * y;
                    });
            }
        }
    }
    (0..n).map(|i| a[i][i]).fold(f64::MIN, f64::max)
}

fn dense<const N: usize>(
    stiffnesses: &ElementNodalStiffnessesSolid<N>,
    masses: &ElementNodalLumpedMasses<N>,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let k = (0..3 * N)
        .map(|i| {
            (0..3 * N)
                .map(|j| stiffnesses[i / 3][j / 3][i % 3][j % 3].value())
                .collect()
        })
        .collect();
    (k, (0..3 * N).map(|i| masses[i / 3].value()).collect())
}

fn hexahedron_problem(
    reference: [[f64; 3]; 8],
    current: [[f64; 3]; 8],
    density: Quantity<Density>,
    material: &NeoHookean,
) -> (ElementNodalStiffnessesSolid<8>, ElementNodalLumpedMasses<8>) {
    let element = Hexahedron::from(ElementNodalReferenceCoordinates::from(reference));
    let stiffnesses = ElasticElement::nodal_stiffnesses(
        &element,
        material,
        &ElementNodalCoordinates::from(current),
    )
    .unwrap();
    let densities: IntegrationDensities<8> = [density; 8].into();
    (
        stiffnesses,
        LumpedMassFiniteElement::nodal_lumped_masses(&element, &densities),
    )
}

fn tetrahedron_problem(
    reference: [[f64; 3]; 4],
    current: [[f64; 3]; 4],
) -> (ElementNodalStiffnessesSolid<4>, ElementNodalLumpedMasses<4>) {
    let element = Tetrahedron::<4>::from(ElementNodalReferenceCoordinates::from(reference));
    let stiffnesses = ElasticElement::nodal_stiffnesses(
        &element,
        &material(),
        &ElementNodalCoordinates::from(current),
    )
    .unwrap();
    let densities: IntegrationDensities<4> = [DENSITY; 4].into();
    (
        stiffnesses,
        LumpedMassFiniteElement::nodal_lumped_masses(&element, &densities),
    )
}

#[test]
fn power_iteration_matches_a_dense_eigensolver_on_a_deformed_hexahedron() {
    let reference = hexahedron_reference();
    let (stiffnesses, masses) =
        hexahedron_problem(reference, deform(reference), DENSITY, &material());
    let (k, m) = dense(&stiffnesses, &masses);
    let exact = jacobi(&k, &m);
    let estimate = largest_eigenvalue(&stiffnesses, &masses);
    assert!(exact > 0.0);
    assert!(estimate <= exact * (1.0 + 1e-9), "{estimate} above {exact}");
    assert!(
        (estimate / exact - 1.0).abs() < 1e-4,
        "{estimate} against {exact}"
    );
}

#[test]
fn power_iteration_matches_a_dense_eigensolver_on_a_deformed_tetrahedron() {
    let reference = tetrahedron_reference();
    let (stiffnesses, masses) = tetrahedron_problem(reference, deform(reference));
    let (k, m) = dense(&stiffnesses, &masses);
    let exact = jacobi(&k, &m);
    let estimate = largest_eigenvalue(&stiffnesses, &masses);
    assert!(estimate <= exact * (1.0 + 1e-9), "{estimate} above {exact}");
    assert!(
        (estimate / exact - 1.0).abs() < 1e-4,
        "{estimate} against {exact}"
    );
}

#[test]
fn the_time_scale_is_the_reciprocal_of_the_highest_frequency() {
    let reference = hexahedron_reference();
    let (stiffnesses, masses) =
        hexahedron_problem(reference, deform(reference), DENSITY, &material());
    let eigenvalue = largest_eigenvalue(&stiffnesses, &masses);
    let scale = fastest_time_scale(&stiffnesses, &masses);
    assert!((scale.value() * eigenvalue.sqrt() - 1.0).abs() < 1e-12);
}

#[test]
fn the_time_scale_grows_with_the_size_of_the_element() {
    let reference = hexahedron_reference();
    let time_scale = |factor: f64| {
        let (stiffnesses, masses) = hexahedron_problem(
            scale(reference, factor),
            scale(deform(reference), factor),
            DENSITY,
            &material(),
        );
        fastest_time_scale(&stiffnesses, &masses).value()
    };
    assert!((time_scale(2.0) / time_scale(1.0) - 2.0).abs() < 1e-6);
}

#[test]
fn the_time_scale_grows_with_the_square_root_of_the_density() {
    let reference = hexahedron_reference();
    let time_scale = |density: Quantity<Density>| {
        let (stiffnesses, masses) =
            hexahedron_problem(reference, deform(reference), density, &material());
        fastest_time_scale(&stiffnesses, &masses).value()
    };
    let ratio = time_scale(4.0 * DENSITY) / time_scale(DENSITY);
    assert!((ratio - 2.0).abs() < 1e-6, "{ratio}");
}

#[test]
fn an_element_without_stiffness_has_an_infinite_time_scale() {
    let reference = hexahedron_reference();
    let (stiffnesses, masses) = hexahedron_problem(
        reference,
        reference,
        DENSITY,
        &NeoHookean {
            shear_modulus: Stress::pascals(0.0),
            bulk_modulus: Stress::pascals(0.0),
        },
    );
    assert!(
        fastest_time_scale(&stiffnesses, &masses)
            .value()
            .is_infinite()
    );
}
