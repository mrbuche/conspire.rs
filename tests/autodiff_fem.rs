#![cfg(all(feature = "fem", feature = "autodiff"))]

use conspire::{
    constitutive::solid::hyperelastic::{NeoHookean, autodiff::AutodiffNeoHookean},
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalReferenceCoordinates,
        linear::Hexahedron,
        solid::{
            elastic::ElasticElement,
            hyperelastic::autodiff::{nodal_forces, nodal_stiffnesses},
        },
    },
    mechanics::DeformationGradient,
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
    let deformation_gradient = DeformationGradient::from([
        [1.04936674, -0.12393166, -0.1092162],
        [0.01618241, 1.08463046, -0.14924521],
        [-0.14722667, 0.03234422, 0.93774805],
    ]);
    let element = Hexahedron::from(reference_coordinates());
    let coordinates = deformation_gradient * reference_coordinates();
    (element, coordinates)
}

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * (1.0 + b.abs())
}

#[test]
fn nodal_forces_match_analytic() {
    let (element, coordinates) = setup();
    let model = AutodiffNeoHookean {
        bulk_modulus: Stress::pascals(BULK_MODULUS),
        shear_modulus: Stress::pascals(SHEAR_MODULUS),
    };
    let hand_model = NeoHookean {
        bulk_modulus: Stress::pascals(BULK_MODULUS),
        shear_modulus: Stress::pascals(SHEAR_MODULUS),
    };
    let autodiff = nodal_forces(&model, &element, &coordinates);
    let hand = element.nodal_forces(&hand_model, &coordinates).unwrap();
    for a in 0..8 {
        for i in 0..3 {
            assert!(
                close(autodiff[3 * a + i], hand[a][i].value(), 1e-8),
                "force [{a}][{i}]"
            );
        }
    }
}

#[test]
fn nodal_stiffnesses_match_analytic() {
    let (element, coordinates) = setup();
    let model = AutodiffNeoHookean {
        bulk_modulus: Stress::pascals(BULK_MODULUS),
        shear_modulus: Stress::pascals(SHEAR_MODULUS),
    };
    let hand_model = NeoHookean {
        bulk_modulus: Stress::pascals(BULK_MODULUS),
        shear_modulus: Stress::pascals(SHEAR_MODULUS),
    };
    let autodiff = nodal_stiffnesses(&model, &element, &coordinates);
    let hand = element
        .nodal_stiffnesses(&hand_model, &coordinates)
        .unwrap();
    for a in 0..8 {
        for b in 0..8 {
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        close(
                            autodiff[3 * a + i][3 * b + j],
                            hand[a][b][i][j].value(),
                            1e-5
                        ),
                        "stiffness [{a}][{b}][{i}][{j}]"
                    );
                }
            }
        }
    }
}
