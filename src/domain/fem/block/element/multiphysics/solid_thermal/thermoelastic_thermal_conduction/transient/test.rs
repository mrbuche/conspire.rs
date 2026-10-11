macro_rules! test_thermoelastic_thermal_conduction_transient_element {
    ($element: ident) => {
        mod thermoelastic_thermal_conduction_transient_element {
            use super::{N, coordinates, element};
            use crate::{
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
                        ElementNodalForcesSolidThermal,
                        thermoelastic_thermal_conduction::{
                            ThermoelasticThermalConductionFiniteElement,
                            transient::ThermoelasticThermalConductionTransientFiniteElement,
                        },
                    },
                    thermal::ElementNodalTemperatures,
                },
                math::assert::{Assert, AssertionError},
                units::Temperature,
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
                        * 100.7;
                    Assert::default().eq_within_tols(
                        &(transient[node_b].1.value() - steady[node_b].1.value()),
                        &expected,
                    )
                })
            }
        }
    };
}
pub(crate) use test_thermoelastic_thermal_conduction_transient_element;
