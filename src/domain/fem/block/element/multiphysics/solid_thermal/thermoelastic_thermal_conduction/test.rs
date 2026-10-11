macro_rules! test_thermoelastic_thermal_conduction_element {
    ($element: ident) => {
        mod thermoelastic_thermal_conduction_element {
            use super::{N, coordinates, element};
            use crate::{
                EPSILON,
                constitutive::{
                    multiphysics::Canonical,
                    solid::thermoelastic::{
                        AlmansiHamel,
                        test::{
                            BULK_MODULUS, COEFFICIENT_OF_THERMAL_EXPANSION, REFERENCE_TEMPERATURE,
                            SHEAR_MODULUS,
                        },
                    },
                    thermal::conduction::{Fourier, test::THERMAL_CONDUCTIVITY},
                },
                fem::block::element::{
                    ElementNodalCoordinates, ElementNodalVelocities, FiniteElementError,
                    multiphysics::solid_thermal::{
                        ElementNodalStiffnessesSolidThermal,
                        ElementNodalForcesSolidThermal,
                        thermoelastic_thermal_conduction::ThermoelasticThermalConductionFiniteElement,
                    },
                    thermal::{ElementNodalTemperatures, conduction::ElementNodalForcesThermal},
                },
                math::assert::{Assert, AssertionError, perturbation},
                math::Tensor,
                units::{Length, Temperature},
            };
            use std::array::from_fn;
            fn model() -> Canonical<AlmansiHamel, Fourier> {
                Canonical::from((
                    AlmansiHamel {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                        coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
                        reference_temperature: REFERENCE_TEMPERATURE,
                    },
                    Fourier {
                        thermal_conductivity: THERMAL_CONDUCTIVITY,
                    },
                ))
            }
            fn temperatures() -> ElementNodalTemperatures<N> {
                (0..N)
                    .map(|node| Temperature::kelvin(100.1 + 0.3 * node as f64))
                    .collect()
            }
            fn forces(
                nodal_coordinates: &ElementNodalCoordinates<N>,
                nodal_temperatures: &ElementNodalTemperatures<N>,
            ) -> Result<ElementNodalForcesSolidThermal<N>, FiniteElementError> {
                ThermoelasticThermalConductionFiniteElement::nodal_forces(
                    &element(),
                    &model(),
                    nodal_coordinates,
                    nodal_temperatures,
                )
            }
            fn stiffnesses(
                nodal_coordinates: &ElementNodalCoordinates<N>,
                nodal_temperatures: &ElementNodalTemperatures<N>,
            ) -> Result<
                ElementNodalStiffnessesSolidThermal<N>,
                FiniteElementError,
            > {
                ThermoelasticThermalConductionFiniteElement::nodal_stiffnesses(
                    &element(),
                    &model(),
                    nodal_coordinates,
                    nodal_temperatures,
                )
            }
            fn forces_stepped_in_coordinate(
                node: usize,
                direction: usize,
            ) -> Result<
                (
                    ElementNodalForcesSolidThermal<N>,
                    ElementNodalForcesSolidThermal<N>,
                ),
                FiniteElementError,
            > {
                let mut nodal_coordinates = coordinates();
                nodal_coordinates[node][direction] += perturbation::<Length>(0.5 * EPSILON);
                let forward = forces(&nodal_coordinates, &temperatures())?;
                nodal_coordinates[node][direction] -= perturbation::<Length>(EPSILON);
                let backward = forces(&nodal_coordinates, &temperatures())?;
                Ok((forward, backward))
            }
            fn forces_stepped_in_temperature(
                node: usize,
            ) -> Result<
                (
                    ElementNodalForcesSolidThermal<N>,
                    ElementNodalForcesSolidThermal<N>,
                ),
                FiniteElementError,
            > {
                let mut nodal_temperatures = temperatures();
                nodal_temperatures[node] += perturbation::<Temperature>(0.5 * EPSILON);
                let forward = forces(&coordinates(), &nodal_temperatures)?;
                nodal_temperatures[node] -= perturbation::<Temperature>(EPSILON);
                let backward = forces(&coordinates(), &nodal_temperatures)?;
                Ok((forward, backward))
            }
            fn velocities() -> ElementNodalVelocities<N> {
                from_fn::<[f64; _], _, _>(|node| {
                    [
                        0.1 + 0.03 * node as f64,
                        -0.2 + 0.05 * node as f64,
                        0.07 - 0.02 * node as f64,
                    ]
                }).into()
            }
            fn heatings(
                nodal_temperatures: &ElementNodalTemperatures<N>,
                nodal_velocities: &ElementNodalVelocities<N>,
            ) -> Result<ElementNodalForcesThermal<N>, FiniteElementError> {
                ThermoelasticThermalConductionFiniteElement::nodal_thermoelastic_heatings(
                    &element(),
                    &model(),
                    &coordinates(),
                    nodal_temperatures,
                    nodal_velocities,
                )
            }
            #[test]
            fn heating_vanishes_without_motion() -> Result<(), AssertionError> {
                let at_rest = ElementNodalVelocities::from(from_fn(|_| [0.0; 3]));
                heatings(&temperatures(), &at_rest)?
                    .iter()
                    .try_for_each(Assert::zero)
            }
            #[test]
            fn heating_follows_coupling_stiffness() -> Result<(), AssertionError> {
                let uniform = (0..N)
                    .map(|_| Temperature::kelvin(100.7))
                    .collect();
                let (_, coupling, _) = stiffnesses(&coordinates(), &uniform)?;
                let velocities = velocities();
                let heatings = heatings(&uniform, &velocities)?;
                assert!(heatings.iter().any(|heating| heating.value() != 0.0));
                (0..N).try_for_each(|node_b| {
                    let expected = (0..N)
                        .map(|node_a| {
                            (0..3)
                                .map(|i| {
                                    coupling[node_a][node_b][i].value()
                                        * velocities[node_a][i].value()
                                })
                                .sum::<f64>()
                        })
                        .sum::<f64>()
                        * 100.7;
                    Assert::default().eq_within_tols(&heatings[node_b].value(), &expected)
                })
            }
            #[test]
            fn solid_stiffness() -> Result<(), AssertionError> {
                let (solid, _, _) = stiffnesses(&coordinates(), &temperatures())?;
                (0..N).try_for_each(|node_b| {
                    (0..3).try_for_each(|k| {
                        let (forward, backward) = forces_stepped_in_coordinate(node_b, k)?;
                        (0..N).try_for_each(|node_a| {
                            (0..3).try_for_each(|i| {
                                Assert::default().eq_within_fd_tol(
                                    &((forward[node_a].0[i] - backward[node_a].0[i])
                                        / perturbation::<Length>(EPSILON)),
                                    &solid[node_a][node_b][i][k],
                                )
                            })
                        })
                    })
                })
            }
            #[test]
            fn coupling_stiffness() -> Result<(), AssertionError> {
                let (_, coupling, _) = stiffnesses(&coordinates(), &temperatures())?;
                assert!((0..N).any(|node_a| {
                    (0..N).any(|node_b| (0..3).any(|i| coupling[node_a][node_b][i].value() != 0.0))
                }));
                (0..N).try_for_each(|node_b| {
                    let (forward, backward) = forces_stepped_in_temperature(node_b)?;
                    (0..N).try_for_each(|node_a| {
                        (0..3).try_for_each(|i| {
                            Assert::default().eq_within_fd_tol(
                                &((forward[node_a].0[i] - backward[node_a].0[i])
                                    / perturbation::<Temperature>(EPSILON)),
                                &coupling[node_a][node_b][i],
                            )
                        })
                    })
                })
            }
            #[test]
            fn thermal_stiffness() -> Result<(), AssertionError> {
                let (_, _, thermal) = stiffnesses(&coordinates(), &temperatures())?;
                (0..N).try_for_each(|node_b| {
                    let (forward, backward) = forces_stepped_in_temperature(node_b)?;
                    (0..N).try_for_each(|node_a| {
                        Assert::default().eq_within_fd_tol(
                            &((forward[node_a].1 - backward[node_a].1)
                                / perturbation::<Temperature>(EPSILON)),
                            &thermal[node_a][node_b],
                        )
                    })
                })
            }
            #[test]
            fn thermal_force_does_not_depend_on_coordinates() -> Result<(), AssertionError> {
                (0..N).try_for_each(|node_b| {
                    (0..3).try_for_each(|k| {
                        let (forward, backward) = forces_stepped_in_coordinate(node_b, k)?;
                        (0..N).try_for_each(|node_a| {
                            Assert::zero(&(forward[node_a].1 - backward[node_a].1))
                        })
                    })
                })
            }
        }
    };
}
pub(crate) use test_thermoelastic_thermal_conduction_element;
