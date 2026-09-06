use super::{nodal_forces, nodal_stiffnesses};
use crate::{
    constitutive::solid::hyperelastic::{NeoHookean, autodiff::neo_hookean::AutodiffNeoHookean},
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalReferenceCoordinates, linear::Hexahedron,
        solid::elastic::ElasticFiniteElement,
    },
    mechanics::test::get_deformation_gradient,
    units::Stress,
};

const BULK_MODULUS: f64 = 1.3;
const SHEAR_MODULUS: f64 = 0.7;

fn reference_coordinates() -> ElementNodalReferenceCoordinates<8> {
    ElementNodalReferenceCoordinates::from([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
}

fn setup() -> (Hexahedron, ElementNodalCoordinates<8>) {
    let element = Hexahedron::from(reference_coordinates());
    let coordinates = get_deformation_gradient() * reference_coordinates();
    (element, coordinates)
}

fn hand() -> NeoHookean {
    NeoHookean {
        bulk_modulus: Stress::pascals(BULK_MODULUS),
        shear_modulus: Stress::pascals(SHEAR_MODULUS),
    }
}

fn autodiff() -> AutodiffNeoHookean {
    AutodiffNeoHookean {
        bulk_modulus: Stress::pascals(BULK_MODULUS),
        shear_modulus: Stress::pascals(SHEAR_MODULUS),
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

#[test]
fn nodal_forces_match_analytic() {
    let (element, coordinates) = setup();
    let ad = nodal_forces(&autodiff(), &element, &coordinates);
    let hand = match element.nodal_forces(&hand(), &coordinates) {
        Ok(forces) => forces,
        Err(_) => panic!("analytic nodal_forces failed"),
    };
    for a in 0..8 {
        for i in 0..3 {
            let (x, y) = (ad[3 * a + i], hand[a][i].value());
            assert!(close(x, y, 1e-8), "force [{a}][{i}]: {x} vs {y}");
        }
    }
}

#[test]
fn nodal_stiffnesses_match_analytic() {
    let (element, coordinates) = setup();
    let ad = nodal_stiffnesses(&autodiff(), &element, &coordinates);
    let hand = match element.nodal_stiffnesses(&hand(), &coordinates) {
        Ok(stiffnesses) => stiffnesses,
        Err(_) => panic!("analytic nodal_stiffnesses failed"),
    };
    for a in 0..8 {
        for b in 0..8 {
            for i in 0..3 {
                for j in 0..3 {
                    let (x, y) = (ad[3 * a + i][3 * b + j], hand[a][b][i][j].value());
                    assert!(
                        close(x, y, 1e-5),
                        "stiffness [{a}][{b}][{i}][{j}]: {x} vs {y}"
                    );
                }
            }
        }
    }
}
