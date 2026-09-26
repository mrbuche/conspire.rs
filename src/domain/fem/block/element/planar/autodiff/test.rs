// shared helpers and test macro for the planar autodiff element tests

pub const A2: [[f64; 2]; 2] = [[1.1, 0.1], [0.05, 0.9]];
pub const F2: [[f64; 2]; 2] = [[1.04936674, -0.12393166], [0.01618241, 1.08463046]];

pub fn apply2(matrix: &[[f64; 2]; 2], x: &[f64; 2]) -> [f64; 2] {
    [
        matrix[0][0] * x[0] + matrix[0][1] * x[1],
        matrix[1][0] * x[0] + matrix[1][1] * x[1],
    ]
}

macro_rules! planar_elastic_tests {
    ($element:ty, $g:literal, $n:literal) => {
        mod elastic {
            use conspire::{
                constitutive::solid::hyperelastic::{NeoHookean, autodiff::AutodiffNeoHookean},
                fem::block::element::{
                    FiniteElement,
                    planar::{
                        PlanarElasticFiniteElement, PlanarElementNodalCoordinates,
                        PlanarElementNodalReferenceCoordinates,
                    },
                    solid::hyperelastic::autodiff::AutodiffElement,
                },
                math::assert::{Assert, AssertionError},
                units::Stress,
            };
            use $crate::common::{BULK_MODULUS, SHEAR_MODULUS};
            use $crate::planar::{A2, F2, apply2};

            fn setup() -> (
                $element,
                PlanarElementNodalCoordinates<$n>,
                AutodiffNeoHookean,
                NeoHookean,
            ) {
                let parametric = <$element as FiniteElement<$g, 2, $n, $n>>::parametric_reference();
                let mut reference = [[0.0; 2]; $n];
                let mut current = [[0.0; 2]; $n];
                for a in 0..$n {
                    let xi = [parametric[a][0].value(), parametric[a][1].value()];
                    reference[a] = apply2(&A2, &xi);
                    current[a] = apply2(&F2, &reference[a]);
                    current[a][0] += 0.03 * reference[a][0] * reference[a][1];
                    current[a][1] += 0.03 * reference[a][0] * reference[a][1];
                }
                let element = <$element>::from(PlanarElementNodalReferenceCoordinates::<$n>::from(
                    reference,
                ));
                let coordinates = PlanarElementNodalCoordinates::<$n>::from(current);
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
                let hd = PlanarElasticFiniteElement::nodal_forces(&element, &hand, &coordinates)
                    .unwrap();
                Assert::default().eq_within_tols(&ad, &hd)
            }

            #[test]
            fn nodal_stiffnesses_match_analytic() -> Result<(), AssertionError> {
                let (element, coordinates, autodiff, hand) = setup();
                let ad = element.autodiff_nodal_stiffnesses(&autodiff, &coordinates);
                let hd =
                    PlanarElasticFiniteElement::nodal_stiffnesses(&element, &hand, &coordinates)
                        .unwrap();
                Assert::default().eq_within_tols(&ad, &hd)
            }
        }
    };
}
