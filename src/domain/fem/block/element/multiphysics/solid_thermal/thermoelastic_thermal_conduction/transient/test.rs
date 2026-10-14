macro_rules! test_thermoelastic_thermal_conduction_transient_element {
    ($element: ident) => {
        mod thermoelastic_thermal_conduction_transient_element {
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
                    ElementNodalVelocities, FiniteElementError,
                    multiphysics::solid_thermal::{
                        ElementNodalForcesSolidThermal, ElementNodalHeatingsSolidThermal,
                        ElementNodalStiffnessesSolidThermal,
                        thermoelastic_thermal_conduction::{
                            ThermoelasticThermalConductionFiniteElement,
                            transient::ThermoelasticThermalConductionTransientFiniteElement,
                        },
                    },
                    thermal::ElementNodalTemperatures,
                },
                math::assert::{Assert, AssertionError, perturbation},
                units::{Temperature, Velocity},
            };
            use std::array::from_fn;
            fn model() -> Canonical<AlmansiHamel, Fourier> {
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
                    .into()
            }
            fn temperatures() -> ElementNodalTemperatures<N> {
                (0..N)
                    .map(|node| Temperature::kelvin(100.1 + 0.3 * node as f64))
                    .collect()
            }
            fn velocities() -> ElementNodalVelocities<N> {
                from_fn(|node| {
                    [
                        0.1 + 0.03 * node as f64,
                        -0.2 + 0.05 * node as f64,
                        0.07 - 0.02 * node as f64,
                    ]
                })
                .into()
            }
            fn at_rest() -> ElementNodalVelocities<N> {
                from_fn(|_| [0.0; 3]).into()
            }
            fn steady(
                nodal_temperatures: &ElementNodalTemperatures<N>,
            ) -> Result<ElementNodalForcesSolidThermal<N>, FiniteElementError> {
                ThermoelasticThermalConductionFiniteElement::nodal_forces(
                    &element(),
                    &model(),
                    &coordinates(),
                    nodal_temperatures,
                )
            }
            fn transient(
                nodal_velocities: &ElementNodalVelocities<N>,
                nodal_temperatures: &ElementNodalTemperatures<N>,
            ) -> Result<ElementNodalForcesSolidThermal<N>, FiniteElementError> {
                ThermoelasticThermalConductionTransientFiniteElement::nodal_forces(
                    &element(),
                    &model(),
                    &coordinates(),
                    nodal_velocities,
                    nodal_temperatures,
                )
            }
            #[test]
            fn at_rest_recovers_steady_forces() -> Result<(), AssertionError> {
                let steady = steady(&temperatures())?;
                let transient = transient(&at_rest(), &temperatures())?;
                (0..N).try_for_each(|node| {
                    (0..3).try_for_each(|i| {
                        Assert::default().eq_within_tols(&transient[node].0[i], &steady[node].0[i])
                    })?;
                    Assert::default().eq_within_tols(&transient[node].1, &steady[node].1)
                })
            }
            #[test]
            fn motion_does_not_change_the_solid_force() -> Result<(), AssertionError> {
                let steady = steady(&temperatures())?;
                let transient = transient(&velocities(), &temperatures())?;
                (0..N).try_for_each(|node| {
                    (0..3).try_for_each(|i| {
                        Assert::default().eq_within_tols(&transient[node].0[i], &steady[node].0[i])
                    })
                })
            }
            #[test]
            fn heating_follows_coupling_stiffness() -> Result<(), AssertionError> {
                let uniform = (0..N).map(|_| Temperature::kelvin(100.7)).collect();
                let (_, coupling, _) =
                    ThermoelasticThermalConductionFiniteElement::nodal_stiffnesses(
                        &element(),
                        &model(),
                        &coordinates(),
                        &uniform,
                    )?;
                let velocities = velocities();
                let steady = steady(&uniform)?;
                let transient = transient(&velocities, &uniform)?;
                assert!((0..N).any(|node| transient[node].1.value() != steady[node].1.value()));
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
                        * -100.7;
                    Assert::default().eq_within_tols(
                        &(transient[node_b].1.value() - steady[node_b].1.value()),
                        &expected,
                    )
                })
            }
            fn stiffnesses(
                nodal_velocities: &ElementNodalVelocities<N>,
                nodal_temperatures: &ElementNodalTemperatures<N>,
            ) -> Result<
                (
                    ElementNodalStiffnessesSolidThermal<N>,
                    ElementNodalHeatingsSolidThermal<N>,
                ),
                FiniteElementError,
            > {
                ThermoelasticThermalConductionTransientFiniteElement::nodal_stiffnesses(
                    &element(),
                    &model(),
                    &coordinates(),
                    nodal_velocities,
                    nodal_temperatures,
                )
            }
            #[test]
            fn at_rest_recovers_steady_stiffnesses() -> Result<(), AssertionError> {
                let ((solid, coupling, thermal), _) = stiffnesses(&at_rest(), &temperatures())?;
                let (steady_solid, steady_coupling, steady_thermal) =
                    ThermoelasticThermalConductionFiniteElement::nodal_stiffnesses(
                        &element(),
                        &model(),
                        &coordinates(),
                        &temperatures(),
                    )?;
                (0..N).try_for_each(|node_a| {
                    (0..N).try_for_each(|node_b| {
                        (0..3).try_for_each(|i| {
                            (0..3).try_for_each(|k| {
                                Assert::default().eq_within_tols(
                                    &solid[node_a][node_b][i][k],
                                    &steady_solid[node_a][node_b][i][k],
                                )
                            })?;
                            Assert::default().eq_within_tols(
                                &coupling[node_a][node_b][i],
                                &steady_coupling[node_a][node_b][i],
                            )
                        })?;
                        Assert::default().eq_within_tols(
                            &thermal[node_a][node_b],
                            &steady_thermal[node_a][node_b],
                        )
                    })
                })
            }
            #[test]
            fn heating_stiffness_is_the_transposed_coupling() -> Result<(), AssertionError> {
                let uniform = (0..N).map(|_| Temperature::kelvin(100.7)).collect();
                let ((_, coupling, _), heatings) = stiffnesses(&velocities(), &uniform)?;
                assert!((0..N).any(|node_a| {
                    (0..N).any(|node_b| (0..3).any(|i| heatings[node_a][node_b][i].value() != 0.0))
                }));
                (0..N).try_for_each(|node_a| {
                    (0..N).try_for_each(|node_b| {
                        (0..3).try_for_each(|i| {
                            Assert::default().eq_within_tols(
                                &heatings[node_a][node_b][i].value(),
                                &(-100.7 * coupling[node_b][node_a][i].value()),
                            )
                        })
                    })
                })
            }
            #[test]
            fn heating_stiffness() -> Result<(), AssertionError> {
                let (_, heatings) = stiffnesses(&velocities(), &temperatures())?;
                (0..N).try_for_each(|node_b| {
                    (0..3).try_for_each(|k| {
                        let mut nodal_velocities = velocities();
                        nodal_velocities[node_b][k] += perturbation::<Velocity>(0.5 * EPSILON);
                        let forward = transient(&nodal_velocities, &temperatures())?;
                        nodal_velocities[node_b][k] -= perturbation::<Velocity>(EPSILON);
                        let backward = transient(&nodal_velocities, &temperatures())?;
                        (0..N).try_for_each(|node_a| {
                            Assert::default().eq_within_fd_tol(
                                &((forward[node_a].1 - backward[node_a].1)
                                    / perturbation::<Velocity>(EPSILON)),
                                &heatings[node_a][node_b][k],
                            )
                        })
                    })
                })
            }
            #[test]
            fn thermal_stiffness_in_motion() -> Result<(), AssertionError> {
                let ((_, _, thermal), _) = stiffnesses(&velocities(), &temperatures())?;
                (0..N).try_for_each(|node_b| {
                    let mut nodal_temperatures = temperatures();
                    nodal_temperatures[node_b] += perturbation::<Temperature>(0.5 * EPSILON);
                    let forward = transient(&velocities(), &nodal_temperatures)?;
                    nodal_temperatures[node_b] -= perturbation::<Temperature>(EPSILON);
                    let backward = transient(&velocities(), &nodal_temperatures)?;
                    (0..N).try_for_each(|node_a| {
                        Assert::default().eq_within_fd_tol(
                            &((forward[node_a].1 - backward[node_a].1)
                                / perturbation::<Temperature>(EPSILON)),
                            &thermal[node_a][node_b],
                        )
                    })
                })
            }
        }
    };
}
pub(crate) use test_thermoelastic_thermal_conduction_transient_element;
