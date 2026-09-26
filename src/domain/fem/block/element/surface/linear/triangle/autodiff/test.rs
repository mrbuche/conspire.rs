use crate::common::{BULK_MODULUS, SHEAR_MODULUS};
use conspire::{
    constitutive::solid::hyperelastic::{NeoHookean, autodiff::AutodiffNeoHookean},
    fem::block::element::{
        FiniteElement,
        planar::{
            PlanarElasticFiniteElement, PlanarElementNodalCoordinates,
            PlanarElementNodalReferenceCoordinates, Triangle,
        },
        solid::hyperelastic::autodiff::AutodiffElement,
    },
    math::assert::{Assert, AssertionError},
    units::Stress,
};

const A2: [[f64; 2]; 2] = [[1.1, 0.1], [0.05, 0.9]];
const F2: [[f64; 2]; 2] = [[1.04936674, -0.12393166], [0.01618241, 1.08463046]];

fn apply2(matrix: &[[f64; 2]; 2], x: &[f64; 2]) -> [f64; 2] {
    [
        matrix[0][0] * x[0] + matrix[0][1] * x[1],
        matrix[1][0] * x[0] + matrix[1][1] * x[1],
    ]
}

fn setup() -> (
    Triangle,
    PlanarElementNodalCoordinates<3>,
    AutodiffNeoHookean,
    NeoHookean,
) {
    let parametric = <Triangle as FiniteElement<1, 2, 3, 3>>::parametric_reference();
    let mut reference = [[0.0; 2]; 3];
    let mut current = [[0.0; 2]; 3];
    for a in 0..3 {
        let xi = [parametric[a][0].value(), parametric[a][1].value()];
        reference[a] = apply2(&A2, &xi);
        current[a] = apply2(&F2, &reference[a]);
    }
    let element = Triangle::from(PlanarElementNodalReferenceCoordinates::<3>::from(reference));
    let coordinates = PlanarElementNodalCoordinates::<3>::from(current);
    let autodiff = AutodiffNeoHookean {
        bulk_modulus: Stress::pascals(BULK_MODULUS),
        shear_modulus: Stress::pascals(SHEAR_MODULUS),
    };
    let hand = NeoHookean {
        bulk_modulus: Stress::pascals(BULK_MODULUS),
        shear_modulus: Stress::pascals(SHEAR_MODULUS),
    };
    (element, coordinates, autodiff, hand)
}

#[test]
fn nodal_forces_match_analytic() -> Result<(), AssertionError> {
    let (element, coordinates, autodiff, hand) = setup();
    let ad = element.autodiff_nodal_forces(&autodiff, &coordinates);
    let hd = PlanarElasticFiniteElement::nodal_forces(&element, &hand, &coordinates).unwrap();
    Assert::default().eq_within_tols(&ad, &hd)
}

#[test]
fn nodal_stiffnesses_match_analytic() -> Result<(), AssertionError> {
    let (element, coordinates, autodiff, hand) = setup();
    let ad = element.autodiff_nodal_stiffnesses(&autodiff, &coordinates);
    let hd = PlanarElasticFiniteElement::nodal_stiffnesses(&element, &hand, &coordinates).unwrap();
    Assert::default().eq_within_tols(&ad, &hd)
}
