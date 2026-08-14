macro_rules! test_thermal_block {
    ($element: ident) => {
        mod thermal_block {
            use super::{
                D, G, M, N, P, get_connectivity, get_reference_coordinates_block, $element,
            };
            use crate::{
                EPSILON,
                constitutive::thermal::conduction::Fourier,
                fem::{
                    ElementModelError,
                    block::{
                        Block,
                        thermal::{
                            NodalTemperatures,
                            conduction::{NodalForcesThermal, NodalStiffnessesThermal},
                        },
                    },
                    thermal::conduction::ThermalConductionElements,
                },
                math::{Quantity, assert::AssertionError},
            };
            mod finite_difference {
                use super::*;
                use crate::units::Temperature;
                const MODEL: Fourier = Fourier {
                    thermal_conductivity: 1.0,
                };
                const EPSILON_TEMPERATURE: Quantity<Temperature> = Quantity::new(EPSILON);
                #[test]
                fn potential() -> Result<(), AssertionError> {
                    let constitutive_model = MODEL;
                    let block = Block::<Fourier, $element, G, M, N, P>::from((
                        constitutive_model,
                        get_connectivity(),
                        &get_reference_coordinates_block(),
                    ));
                    let mut finite_difference = Quantity::default();
                    let nodal_forces_fd: NodalForcesThermal = (0..D)
                        .map(|node| {
                            let mut nodal_temperatures = NodalTemperatures::zero(D);
                            nodal_temperatures[node] += EPSILON_TEMPERATURE * 0.5;
                            finite_difference = block.potential(&nodal_temperatures)?;
                            nodal_temperatures[node] -= EPSILON_TEMPERATURE;
                            finite_difference -= block.potential(&nodal_temperatures)?;
                            // A potential per unit temperature is a power.
                            Ok(finite_difference / EPSILON_TEMPERATURE)
                        })
                        .collect::<Result<_, ElementModelError>>()?;
                    $crate::math::assert::Assert::default().eq_within_fd_tol(
                        &nodal_forces_fd,
                        &block.nodal_forces(&NodalTemperatures::zero(D))?,
                    )
                }
                #[test]
                fn nodal_forces() -> Result<(), AssertionError> {
                    let constitutive_model = MODEL;
                    let block = Block::<Fourier, $element, G, M, N, P>::from((
                        constitutive_model,
                        get_connectivity(),
                        &get_reference_coordinates_block(),
                    ));
                    let mut finite_difference = Quantity::default();
                    let nodal_stiffnesses_fd: NodalStiffnessesThermal = (0..D)
                        .map(|node_a| {
                            (0..D)
                                .map(|node_b| {
                                    let mut nodal_temperatures = NodalTemperatures::zero(D);
                                    nodal_temperatures[node_b] += EPSILON_TEMPERATURE * 0.5;
                                    finite_difference =
                                        block.nodal_forces(&nodal_temperatures)?[node_a];
                                    nodal_temperatures[node_b] -= EPSILON_TEMPERATURE;
                                    finite_difference -=
                                        block.nodal_forces(&nodal_temperatures)?[node_a];
                                    // A power per unit temperature is a thermal stiffness.
                                    Ok(finite_difference / EPSILON_TEMPERATURE)
                                })
                                .collect()
                        })
                        .collect::<Result<_, ElementModelError>>()?;
                    $crate::math::assert::Assert::default().eq_within_fd_tol(
                        &nodal_stiffnesses_fd,
                        &block.nodal_stiffnesses(&NodalTemperatures::zero(D))?,
                    )
                }
            }
        }
    };
}
pub(crate) use test_thermal_block;
