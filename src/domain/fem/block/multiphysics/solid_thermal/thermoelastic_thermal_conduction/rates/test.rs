macro_rules! test_thermoelastic_thermal_conduction_rate_block {
    ($element: ident) => {
        mod thermoelastic_thermal_conduction_rate_block {
            use super::{
                D, G, M, N, P, get_connectivity, get_coordinates_block,
                get_reference_coordinates_block, get_velocities_block, $element,
            };
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
                fem::{
                    NodalVelocities,
                    block::{Block, thermal::NodalTemperatures},
                    multiphysics::solid_thermal::thermoelastic_thermal_conduction::{
                        rates::ThermoelasticThermalConductionRateElements,
                        steady::ThermoelasticThermalConductionElements,
                    },
                },
                math::{
                    Hessian, Tensor,
                    assert::{Assert, AssertionError, perturbation},
                },
                units::{Length, Temperature, Velocity},
            };
            fn coupled_block() -> Block<Canonical<AlmansiHamel, Fourier>, $element, G, M, N, P> {
                (
                    (
                        AlmansiHamel {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                            coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
                            reference_temperature: REFERENCE_TEMPERATURE,
                        },
                        Fourier {
                            thermal_conductivity: THERMAL_CONDUCTIVITY,
                        },
                    )
                        .into(),
                    get_connectivity(),
                    &get_reference_coordinates_block(),
                )
                    .into()
            }
            fn nonuniform_temperatures() -> NodalTemperatures {
                (0..D)
                    .map(|node| Temperature::kelvin(100.0 + 0.7 * (node as f64).sin() + 0.1))
                    .collect()
            }
            fn uniform_temperatures(temperature: f64) -> NodalTemperatures {
                (0..D).map(|_| Temperature::kelvin(temperature)).collect()
            }
            fn at_rest() -> NodalVelocities<3> {
                (0..D).map(|_| Default::default()).collect()
            }
            #[test]
            fn at_rest_recovers_steady_forces() -> Result<(), AssertionError> {
                let block = coupled_block();
                let temperatures = nonuniform_temperatures();
                let steady = ThermoelasticThermalConductionElements::nodal_forces(
                    &block,
                    &get_coordinates_block(),
                    &temperatures,
                )?;
                let transient = ThermoelasticThermalConductionRateElements::nodal_forces(
                    &block,
                    &get_coordinates_block(),
                    &at_rest(),
                    &temperatures,
                )?;
                assert!(steady.iter().any(|force| force.1.value() != 0.0));
                steady
                    .iter()
                    .zip(transient.iter())
                    .try_for_each(|(steady, transient)| {
                        (0..3).try_for_each(|i| {
                            Assert::default().eq_within_fd_tol(&steady.0[i], &transient.0[i])
                        })?;
                        Assert::default().eq_within_fd_tol(&steady.1, &transient.1)
                    })
            }
            #[test]
            fn heating_follows_coupling_stiffness() -> Result<(), AssertionError> {
                let block = coupled_block();
                let temperature = 100.7;
                let uniform = uniform_temperatures(temperature);
                let velocities = get_velocities_block();
                let stiffnesses = ThermoelasticThermalConductionElements::nodal_stiffnesses(
                    &block,
                    &get_coordinates_block(),
                    &uniform,
                )?;
                let steady = ThermoelasticThermalConductionElements::nodal_forces(
                    &block,
                    &get_coordinates_block(),
                    &uniform,
                )?;
                let transient = ThermoelasticThermalConductionRateElements::nodal_forces(
                    &block,
                    &get_coordinates_block(),
                    &velocities,
                    &uniform,
                )?;
                assert!(
                    steady
                        .iter()
                        .zip(transient.iter())
                        .any(|(steady, transient)| steady.1.value() != transient.1.value())
                );
                (0..D).try_for_each(|node_b| {
                    let expected = (0..D)
                        .map(|node_a| {
                            (0..3)
                                .map(|i| {
                                    stiffnesses.entry(4 * node_a + i, 4 * node_b + 3)
                                        * velocities[node_a][i].value()
                                })
                                .sum::<f64>()
                        })
                        .sum::<f64>()
                        * -temperature;
                    Assert::default().eq_within_fd_tol(
                        &(transient[node_b].1.value() - steady[node_b].1.value()),
                        &expected,
                    )
                })
            }
            #[test]
            fn the_heating_stiffnesses_match_finite_differences() -> Result<(), AssertionError> {
                let block = coupled_block();
                let temperatures = nonuniform_temperatures();
                let velocities = get_velocities_block();
                let (stiffnesses, heatings) =
                    ThermoelasticThermalConductionRateElements::nodal_stiffnesses(
                        &block,
                        &get_coordinates_block(),
                        &velocities,
                        &temperatures,
                    )?;
                let heating = |a: usize, b: usize, k: usize| {
                    heatings[a]
                        .entries()
                        .find(|&(column, _)| column == b)
                        .map_or(0.0, |(_, heating)| heating[k].value())
                };
                let mut any = false;
                (0..D).try_for_each(|node_b| {
                    (0..3).try_for_each(|k| {
                        let mut stepped = velocities.clone();
                        stepped[node_b][k] += perturbation::<Velocity>(0.5 * EPSILON);
                        let forward = ThermoelasticThermalConductionRateElements::nodal_forces(
                            &block,
                            &get_coordinates_block(),
                            &stepped,
                            &temperatures,
                        )?;
                        stepped[node_b][k] -= perturbation::<Velocity>(EPSILON);
                        let backward = ThermoelasticThermalConductionRateElements::nodal_forces(
                            &block,
                            &get_coordinates_block(),
                            &stepped,
                            &temperatures,
                        )?;
                        (0..D).try_for_each(|node_a| {
                            let expected =
                                (forward[node_a].1 - backward[node_a].1).value() / EPSILON;
                            any |= expected != 0.0;
                            Assert::default()
                                .eq_within_fd_tol(&expected, &heating(node_a, node_b, k))
                        })
                    })
                })?;
                assert!(any);
                let mut any = false;
                (0..D).try_for_each(|node_b| {
                    (0..3).try_for_each(|k| {
                        let mut stepped = get_coordinates_block();
                        stepped[node_b][k] += perturbation::<Length>(0.5 * EPSILON);
                        let forward = ThermoelasticThermalConductionRateElements::nodal_forces(
                            &block,
                            &stepped,
                            &velocities,
                            &temperatures,
                        )?;
                        stepped[node_b][k] -= perturbation::<Length>(EPSILON);
                        let backward = ThermoelasticThermalConductionRateElements::nodal_forces(
                            &block,
                            &stepped,
                            &velocities,
                            &temperatures,
                        )?;
                        (0..D).try_for_each(|node_a| {
                            let expected =
                                (forward[node_a].1 - backward[node_a].1).value() / EPSILON;
                            any |= expected != 0.0;
                            Assert::default().eq_within_fd_tol(
                                &expected,
                                &stiffnesses.entry(4 * node_a + 3, 4 * node_b + k),
                            )
                        })
                    })
                })?;
                assert!(any);
                Ok(())
            }
        }
    };
}
pub(crate) use test_thermoelastic_thermal_conduction_rate_block;
