#![cfg(all(feature = "fem", feature = "autodiff"))]

use conspire::{
    constitutive::{
        canonical::Canonical,
        fluid::hyperviscous::{Newtonian, autodiff::AutodiffNewtonian},
        solid::hyperelastic::{
            NeoHookean,
            autodiff::{Autodiff, AutodiffNeoHookean},
        },
    },
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalReferenceCoordinates, ElementNodalVelocities,
        FiniteElement, linear,
        planar::{
            PlanarElasticFiniteElement, PlanarElementNodalCoordinates,
            PlanarElementNodalReferenceCoordinates, Triangle,
        },
        solid::{
            elastic::ElasticElement,
            elastic_hyperviscous::ElasticHyperviscousElement,
            hyperelastic::autodiff::{AutodiffElement, AutodiffViscoelasticElement},
            hyperviscoelastic::HyperviscoelasticElement,
            viscoelastic::ViscoelasticElement,
        },
    },
    units::{Stress, Viscosity},
};

const BULK_MODULUS: f64 = 1.3;
const SHEAR_MODULUS: f64 = 0.7;

const A: [[f64; 3]; 3] = [[1.1, 0.1, 0.0], [0.05, 0.9, 0.1], [0.0, 0.1, 1.2]];
const F: [[f64; 3]; 3] = [
    [1.04936674, -0.12393166, -0.1092162],
    [0.01618241, 1.08463046, -0.14924521],
    [-0.14722667, 0.03234422, 0.93774805],
];

fn apply(matrix: &[[f64; 3]; 3], x: &[f64; 3]) -> [f64; 3] {
    let mut y = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            y[i] += matrix[i][j] * x[j];
        }
    }
    y
}

fn reference<const N: usize>(parametric: [[f64; 3]; N]) -> [[f64; 3]; N] {
    parametric.map(|xi| apply(&A, &xi))
}

fn deformed<const N: usize>(reference: &[[f64; 3]; N]) -> [[f64; 3]; N] {
    reference.map(|x| {
        let mut y = apply(&F, &x);
        y[0] += 0.03 * x[1] * x[2];
        y[1] += 0.03 * x[2] * x[0];
        y[2] += 0.03 * x[0] * x[1];
        y
    })
}

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * (1.0 + b.abs())
}

macro_rules! shape {
    ($name:ident, $element:ty, $g:literal, $n:literal) => {
        mod $name {
            use super::*;

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
            fn nodal_forces_match_analytic() {
                let (element, coordinates, autodiff, hand) = setup();
                let ad = element.autodiff_nodal_forces(&autodiff, &coordinates);
                let hd = ElasticElement::nodal_forces(&element, &hand, &coordinates).unwrap();
                for a in 0..$n {
                    for i in 0..3 {
                        assert!(
                            close(ad[a][i].value(), hd[a][i].value(), 1e-8),
                            "force [{a}][{i}]"
                        );
                    }
                }
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
                    println!("BENCH {} {label}: {micros:.2} us", stringify!($name));
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
            fn nodal_stiffnesses_match_analytic() {
                let (element, coordinates, autodiff, hand) = setup();
                let ad = element.autodiff_nodal_stiffnesses(&autodiff, &coordinates);
                let hd = ElasticElement::nodal_stiffnesses(&element, &hand, &coordinates).unwrap();
                for a in 0..$n {
                    for b in 0..$n {
                        for i in 0..3 {
                            for j in 0..3 {
                                assert!(
                                    close(ad[a][b][i][j].value(), hd[a][b][i][j].value(), 1e-5),
                                    "stiffness [{a}][{b}][{i}][{j}]"
                                );
                            }
                        }
                    }
                }
            }
        }
    };
}

shape!(linear_hexahedron, linear::Hexahedron, 8, 8);
shape!(linear_tetrahedron, linear::Tetrahedron, 1, 4);

mod linear_triangle {
    use super::*;

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
    fn nodal_forces_match_analytic() {
        let (element, coordinates, autodiff, hand) = setup();
        let ad = element.autodiff_nodal_forces(&autodiff, &coordinates);
        let hd = PlanarElasticFiniteElement::nodal_forces(&element, &hand, &coordinates).unwrap();
        for a in 0..3 {
            for i in 0..2 {
                assert!(
                    close(ad[a][i].value(), hd[a][i].value(), 1e-8),
                    "force [{a}][{i}]"
                );
            }
        }
    }

    #[test]
    fn nodal_stiffnesses_match_analytic() {
        let (element, coordinates, autodiff, hand) = setup();
        let ad = element.autodiff_nodal_stiffnesses(&autodiff, &coordinates);
        let hd =
            PlanarElasticFiniteElement::nodal_stiffnesses(&element, &hand, &coordinates).unwrap();
        for a in 0..3 {
            for b in 0..3 {
                for i in 0..2 {
                    for j in 0..2 {
                        assert!(
                            close(ad[a][b][i][j].value(), hd[a][b][i][j].value(), 1e-5),
                            "stiffness [{a}][{b}][{i}][{j}]"
                        );
                    }
                }
            }
        }
    }
}

const L: [[f64; 3]; 3] = [
    [0.05, -0.02, 0.01],
    [0.03, 0.04, -0.06],
    [-0.01, 0.02, 0.07],
];

macro_rules! viscous_shape {
    ($name:ident, $element:ty, $g:literal, $n:literal) => {
        mod $name {
            use super::*;

            #[allow(clippy::type_complexity)]
            fn setup() -> (
                $element,
                ElementNodalCoordinates<$n>,
                ElementNodalVelocities<$n>,
                Canonical<Autodiff<AutodiffNeoHookean>, Autodiff<AutodiffNewtonian>>,
                Canonical<NeoHookean, Newtonian>,
            ) {
                let parametric = <$element as FiniteElement<$g, 3, $n, $n>>::parametric_reference();
                let mut nodes = [[0.0; 3]; $n];
                for a in 0..$n {
                    for k in 0..3 {
                        nodes[a][k] = parametric[a][k].value();
                    }
                }
                let reference = reference(nodes);
                let mut velocities = [[0.0; 3]; $n];
                for a in 0..$n {
                    let x = reference[a];
                    velocities[a] = apply(&L, &x);
                    velocities[a][0] += 0.02 * x[1] * x[2];
                    velocities[a][1] += 0.02 * x[2] * x[0];
                    velocities[a][2] += 0.02 * x[0] * x[1];
                }
                let element =
                    <$element>::from(ElementNodalReferenceCoordinates::<$n>::from(reference));
                let coordinates = ElementNodalCoordinates::<$n>::from(deformed(&reference));
                let velocities = ElementNodalVelocities::<$n>::from(velocities);
                let (bulk_modulus, shear_modulus) = (
                    Stress::pascals(BULK_MODULUS),
                    Stress::pascals(SHEAR_MODULUS),
                );
                let (bulk_viscosity, shear_viscosity) = (
                    Viscosity::pascal_seconds(1.1),
                    Viscosity::pascal_seconds(0.5),
                );
                let autodiff = Canonical::from((
                    Autodiff(AutodiffNeoHookean {
                        bulk_modulus,
                        shear_modulus,
                    }),
                    Autodiff(AutodiffNewtonian {
                        bulk_viscosity,
                        shear_viscosity,
                    }),
                ));
                let hand = Canonical::from((
                    NeoHookean {
                        bulk_modulus,
                        shear_modulus,
                    },
                    Newtonian {
                        bulk_viscosity,
                        shear_viscosity,
                    },
                ));
                (element, coordinates, velocities, autodiff, hand)
            }

            #[test]
            fn nodal_forces_match_analytic() {
                let (element, coordinates, velocities, autodiff, hand) = setup();
                let ad = element.autodiff_viscoelastic_nodal_forces(
                    &autodiff,
                    &coordinates,
                    &velocities,
                );
                let hd =
                    ViscoelasticElement::nodal_forces(&element, &hand, &coordinates, &velocities)
                        .unwrap();
                for a in 0..$n {
                    for i in 0..3 {
                        assert!(
                            close(ad[a][i].value(), hd[a][i].value(), 1e-8),
                            "force [{a}][{i}]"
                        );
                    }
                }
            }

            #[test]
            fn nodal_dampings_match_analytic() {
                let (element, coordinates, velocities, autodiff, hand) = setup();
                let ad = element.autodiff_nodal_dampings(&autodiff, &coordinates, &velocities);
                let hd = ViscoelasticElement::nodal_stiffnesses(
                    &element,
                    &hand,
                    &coordinates,
                    &velocities,
                )
                .unwrap();
                for a in 0..$n {
                    for b in 0..$n {
                        for i in 0..3 {
                            for j in 0..3 {
                                assert!(
                                    close(ad[a][b][i][j].value(), hd[a][b][i][j].value(), 1e-6),
                                    "damping [{a}][{b}][{i}][{j}]"
                                );
                            }
                        }
                    }
                }
            }

            #[test]
            fn energies_match_analytic() {
                let (element, coordinates, velocities, autodiff, hand) = setup();
                let ad = element.autodiff_viscous_dissipation(&autodiff, &coordinates, &velocities);
                let hd = ElasticHyperviscousElement::viscous_dissipation(
                    &element,
                    &hand,
                    &coordinates,
                    &velocities,
                )
                .unwrap();
                assert!(close(ad.value(), hd.value(), 1e-10), "viscous dissipation");
                let ad = element.autodiff_helmholtz_free_energy(&autodiff, &coordinates);
                let hd =
                    HyperviscoelasticElement::helmholtz_free_energy(&element, &hand, &coordinates)
                        .unwrap();
                assert!(
                    close(ad.value(), hd.value(), 1e-10),
                    "helmholtz free energy"
                );
            }
        }
    };
}

viscous_shape!(viscous_hexahedron, linear::Hexahedron, 8, 8);
viscous_shape!(viscous_tetrahedron, linear::Tetrahedron, 1, 4);
