macro_rules! test_thermoelastic_thermal_conduction_block {
    ($element: ident) => {
        mod thermoelastic_thermal_conduction_block {
            use super::{
                D, G, M, N, P, get_connectivity, get_coordinates_block,
                get_reference_coordinates_block, $element,
            };
            use crate::{
                EPSILON,
                domain::diffusion::DiffusionElements,
                constitutive::{
                    multiphysics::Canonical,
                    solid::{
                        elastic::AlmansiHamelEulerian,
                        thermoelastic::{
                            AlmansiHamel,
                            test::{
                                BULK_MODULUS, COEFFICIENT_OF_THERMAL_EXPANSION,
                                REFERENCE_TEMPERATURE, SHEAR_MODULUS,
                            },
                        },
                    },
                    thermal::conduction::{Fourier, test::THERMAL_CONDUCTIVITY},
                },
                fem::{
                    block::{Block, thermal::NodalTemperatures},
                    multiphysics::solid_thermal::thermoelastic_thermal_conduction::steady::ThermoelasticThermalConductionElements,
                    solid::elastic::ElasticElements,
                },
                math::{
                    Hessian, Quantity, Tensor,
                    assert::{Assert, AssertionError, perturbation},
                },
                units::{Length, Power, Temperature},
            };
            fn coupled_block() -> Block<
                Canonical<AlmansiHamel, Fourier>,
                $element,
                G,
                M,
                N,
                P,
            > {
                Block::from((
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
                    )),
                    get_connectivity(),
                    &get_reference_coordinates_block(),
                ))
            }
            fn nonuniform_temperatures() -> NodalTemperatures {
                (0..D)
                    .map(|node| Temperature::kelvin(100.0 + 0.7 * (node as f64).sin() + 0.1))
                    .collect()
            }
            fn uniform_temperatures(temperature: f64) -> NodalTemperatures {
                (0..D).map(|_| Temperature::kelvin(temperature)).collect()
            }
            #[test]
            fn thermal_rows_are_the_conduction_forces() -> Result<(), AssertionError> {
                let temperatures = nonuniform_temperatures();
                let thermal_block = Block::<Fourier, $element, G, M, N, P>::from((
                    Fourier {
                        thermal_conductivity: THERMAL_CONDUCTIVITY,
                    },
                    get_connectivity(),
                    &get_reference_coordinates_block(),
                ));
                let thermal_forces = DiffusionElements::<Power>::nodal_forces(&thermal_block, &temperatures)?;
                let forces = ThermoelasticThermalConductionElements::nodal_forces(
                    &coupled_block(),
                    &get_coordinates_block(),
                    &temperatures,
                )?;
                assert!(thermal_forces.iter().any(|force| force.value() != 0.0));
                thermal_forces
                    .iter()
                    .zip(forces.iter())
                    .try_for_each(|(thermal_force, force)| {
                        Assert::default().eq_within_fd_tol(thermal_force, &force.1)
                    })
            }
            #[test]
            fn solid_rows_are_the_elastic_forces_at_the_reference_temperature()
            -> Result<(), AssertionError> {
                let elastic_block = Block::<AlmansiHamelEulerian, $element, G, M, N, P>::from((
                    AlmansiHamelEulerian {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                    get_connectivity(),
                    &get_reference_coordinates_block(),
                ));
                let elastic_forces = ElasticElements::<3>::nodal_forces(&elastic_block, &get_coordinates_block())?;
                let forces = ThermoelasticThermalConductionElements::nodal_forces(
                    &coupled_block(),
                    &get_coordinates_block(),
                    &uniform_temperatures(REFERENCE_TEMPERATURE.value()),
                )?;
                elastic_forces
                    .iter()
                    .zip(forces.iter())
                    .try_for_each(|(elastic_force, force)| {
                        Assert::default().eq_within_fd_tol(elastic_force, &force.0)
                    })
            }
            #[test]
            fn solid_rows_respond_to_temperature() {
                let block = coupled_block();
                let reference = ThermoelasticThermalConductionElements::nodal_forces(
                    &block,
                    &get_coordinates_block(),
                    &uniform_temperatures(REFERENCE_TEMPERATURE.value()),
                )
                .unwrap();
                let heated = ThermoelasticThermalConductionElements::nodal_forces(
                    &block,
                    &get_coordinates_block(),
                    &uniform_temperatures(REFERENCE_TEMPERATURE.value() + 10.0),
                )
                .unwrap();
                assert!(
                    reference
                        .iter()
                        .zip(heated.iter())
                        .any(|(a, b)| a.0 != b.0)
                )
            }
            #[test]
            fn uniform_temperature_has_no_thermal_rows() -> Result<(), AssertionError> {
                ThermoelasticThermalConductionElements::nodal_forces(
                    &coupled_block(),
                    &get_coordinates_block(),
                    &uniform_temperatures(250.0),
                )?
                .iter()
                    .try_for_each(|force| Assert::default().eq_within_fd_tol(&Quantity::<Power>::default(), &force.1))
            }
            #[test]
            fn nodal_stiffnesses_match_finite_differences() -> Result<(), AssertionError> {
                let block = coupled_block();
                let coordinates = get_coordinates_block();
                let temperatures = nonuniform_temperatures();
                let stiffnesses = ThermoelasticThermalConductionElements::nodal_stiffnesses(
                    &block,
                    &coordinates,
                    &temperatures,
                )?;
                (0..D).try_for_each(|node_b| {
                    (0..4).try_for_each(|dof| {
                        let (forward, backward) = if dof < 3 {
                            let mut stepped = coordinates.clone();
                            stepped[node_b][dof] += perturbation::<Length>(0.5 * EPSILON);
                            let forward = ThermoelasticThermalConductionElements::nodal_forces(
                                &block,
                                &stepped,
                                &temperatures,
                            )?;
                            stepped[node_b][dof] -= perturbation::<Length>(EPSILON);
                            let backward = ThermoelasticThermalConductionElements::nodal_forces(
                                &block,
                                &stepped,
                                &temperatures,
                            )?;
                            (forward, backward)
                        } else {
                            let mut stepped = temperatures.clone();
                            stepped[node_b] += perturbation::<Temperature>(0.5 * EPSILON);
                            let forward = ThermoelasticThermalConductionElements::nodal_forces(
                                &block,
                                &coordinates,
                                &stepped,
                            )?;
                            stepped[node_b] -= perturbation::<Temperature>(EPSILON);
                            let backward = ThermoelasticThermalConductionElements::nodal_forces(
                                &block,
                                &coordinates,
                                &stepped,
                            )?;
                            (forward, backward)
                        };
                        (0..D).try_for_each(|node_a| {
                            (0..4).try_for_each(|row| {
                                let difference = if row < 3 {
                                    (forward[node_a].0[row] - backward[node_a].0[row]).value()
                                } else {
                                    (forward[node_a].1 - backward[node_a].1).value()
                                };
                                Assert::default().eq_within_fd_tol(
                                    &(difference / EPSILON),
                                    &stiffnesses.entry(4 * node_a + row, 4 * node_b + dof),
                                )
                            })
                        })
                    })
                })
            }
        }
    };
}
pub(crate) use test_thermoelastic_thermal_conduction_block;
