// shared helpers and test macros for the autodiff element tests

pub const BULK_MODULUS: f64 = 1.3;
pub const SHEAR_MODULUS: f64 = 0.7;

const A: [[f64; 3]; 3] = [[1.1, 0.1, 0.0], [0.05, 0.9, 0.1], [0.0, 0.1, 1.2]];
const F: [[f64; 3]; 3] = [
    [1.04936674, -0.12393166, -0.1092162],
    [0.01618241, 1.08463046, -0.14924521],
    [-0.14722667, 0.03234422, 0.93774805],
];

pub fn apply(matrix: &[[f64; 3]; 3], x: &[f64; 3]) -> [f64; 3] {
    let mut y = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            y[i] += matrix[i][j] * x[j];
        }
    }
    y
}

pub fn reference<const N: usize>(parametric: [[f64; 3]; N]) -> [[f64; 3]; N] {
    parametric.map(|xi| apply(&A, &xi))
}

pub fn deformed<const N: usize>(reference: &[[f64; 3]; N]) -> [[f64; 3]; N] {
    reference.map(|x| {
        let mut y = apply(&F, &x);
        y[0] += 0.03 * x[1] * x[2];
        y[1] += 0.03 * x[2] * x[0];
        y[2] += 0.03 * x[0] * x[1];
        y
    })
}

macro_rules! elastic_tests {
    ($element:ty, $g:literal, $n:literal) => {
        mod elastic {
            use conspire::{
                constitutive::solid::hyperelastic::{NeoHookean, autodiff::AutodiffNeoHookean},
                fem::block::element::{
                    ElementNodalCoordinates, ElementNodalReferenceCoordinates, FiniteElement,
                    solid::{elastic::ElasticElement, hyperelastic::autodiff::AutodiffElement},
                },
                math::assert::{Assert, AssertionError},
                units::Stress,
            };
            use $crate::common::{BULK_MODULUS, SHEAR_MODULUS, deformed, reference};

            fn setup() -> (
                $element,
                ElementNodalCoordinates<$n>,
                AutodiffNeoHookean,
                NeoHookean,
            ) {
                let parametric = <$element as FiniteElement<$g, 3, $n, $n>>::parametric_reference();
                let mut nodes = [[0.0; 3]; $n];
                for a in 0..$n {
                    for k in 0..3 {
                        nodes[a][k] = parametric[a][k].value();
                    }
                }
                let reference = reference(nodes);
                let element =
                    <$element>::from(ElementNodalReferenceCoordinates::<$n>::from(reference));
                let coordinates = ElementNodalCoordinates::<$n>::from(deformed(&reference));
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
                let hd = ElasticElement::nodal_forces(&element, &hand, &coordinates).unwrap();
                Assert::default().eq_within_tols(&ad, &hd)
            }

            #[test]
            #[ignore]
            fn bench() {
                use std::{hint::black_box, time::Instant};
                let (element, coordinates, autodiff, hand) = setup();
                let time = |label: &str, iterations: u32, mut f: Box<dyn FnMut() + '_>| {
                    f();
                    let start = Instant::now();
                    for _ in 0..iterations {
                        f();
                    }
                    let micros = start.elapsed().as_secs_f64() * 1e6 / iterations as f64;
                    println!("BENCH {} {label}: {micros:.2} us", stringify!($element));
                };
                time(
                    "forces autodiff",
                    20000,
                    Box::new(|| {
                        black_box(
                            element.autodiff_nodal_forces(&autodiff, black_box(&coordinates)),
                        );
                    }),
                );
                time(
                    "forces hand    ",
                    20000,
                    Box::new(|| {
                        black_box(
                            ElasticElement::nodal_forces(&element, &hand, black_box(&coordinates))
                                .unwrap(),
                        );
                    }),
                );
                time(
                    "stiffness autodiff",
                    2000,
                    Box::new(|| {
                        black_box(
                            element.autodiff_nodal_stiffnesses(&autodiff, black_box(&coordinates)),
                        );
                    }),
                );
                time(
                    "stiffness hand    ",
                    2000,
                    Box::new(|| {
                        black_box(
                            ElasticElement::nodal_stiffnesses(
                                &element,
                                &hand,
                                black_box(&coordinates),
                            )
                            .unwrap(),
                        );
                    }),
                );
            }

            #[test]
            fn nodal_stiffnesses_match_analytic() -> Result<(), AssertionError> {
                let (element, coordinates, autodiff, hand) = setup();
                let ad = element.autodiff_nodal_stiffnesses(&autodiff, &coordinates);
                let hd = ElasticElement::nodal_stiffnesses(&element, &hand, &coordinates).unwrap();
                Assert::default().eq_within_tols(&ad, &hd)
            }
        }
    };
}
