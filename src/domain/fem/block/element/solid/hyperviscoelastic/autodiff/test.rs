// tests for the autodiff hyperviscoelastic elements

pub const L: [[f64; 3]; 3] = [
    [0.05, -0.02, 0.01],
    [0.03, 0.04, -0.06],
    [-0.01, 0.02, 0.07],
];

macro_rules! viscous_tests {
    ($element:ty, $g:literal, $n:literal) => {
        mod viscous {
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
                    ElementNodalCoordinates, ElementNodalReferenceCoordinates,
                    ElementNodalVelocities, FiniteElement,
                    solid::{
                        elastic_hyperviscous::ElasticHyperviscousElement,
                        hyperviscoelastic::{
                            HyperviscoelasticElement, autodiff::AutodiffViscoelasticElement,
                        },
                        viscoelastic::ViscoelasticElement,
                    },
                },
                math::assert::{Assert, AssertionError},
                units::{Stress, Viscosity},
            };
            use $crate::common::{BULK_MODULUS, SHEAR_MODULUS, apply, deformed, reference};
            use $crate::hyperviscoelastic::L;

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
            fn nodal_forces_match_analytic() -> Result<(), AssertionError> {
                let (element, coordinates, velocities, autodiff, hand) = setup();
                let ad = element.autodiff_viscoelastic_nodal_forces(
                    &autodiff,
                    &coordinates,
                    &velocities,
                );
                let hd =
                    ViscoelasticElement::nodal_forces(&element, &hand, &coordinates, &velocities)
                        .unwrap();
                Assert::default().eq_within_tols(&ad, &hd)
            }

            #[test]
            fn nodal_dampings_match_analytic() -> Result<(), AssertionError> {
                let (element, coordinates, velocities, autodiff, hand) = setup();
                let ad = element.autodiff_nodal_dampings(&autodiff, &coordinates, &velocities);
                let hd = ViscoelasticElement::nodal_stiffnesses(
                    &element,
                    &hand,
                    &coordinates,
                    &velocities,
                )
                .unwrap();
                Assert::default().eq_within_tols(&ad, &hd)
            }

            #[test]
            fn energies_match_analytic() -> Result<(), AssertionError> {
                let (element, coordinates, velocities, autodiff, hand) = setup();
                let ad = element.autodiff_viscous_dissipation(&autodiff, &coordinates, &velocities);
                let hd = ElasticHyperviscousElement::viscous_dissipation(
                    &element,
                    &hand,
                    &coordinates,
                    &velocities,
                )
                .unwrap();
                Assert::default().eq_within_tols(&ad, &hd)?;
                let ad = element.autodiff_helmholtz_free_energy(&autodiff, &coordinates);
                let hd =
                    HyperviscoelasticElement::helmholtz_free_energy(&element, &hand, &coordinates)
                        .unwrap();
                Assert::default().eq_within_tols(&ad, &hd)
            }
        }
    };
}
