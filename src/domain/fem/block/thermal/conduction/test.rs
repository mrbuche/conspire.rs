macro_rules! test_thermal {
    ($element: ident) => {
        mod thermal_block {
            use super::{D, G, M, N, P, get_connectivity, get_reference_coordinates_block, $element};
            use crate::{
                EPSILON,
                constitutive::thermal::conduction::Fourier,
                fem::block::{
                    Block, FiniteElementBlockError,
                    thermal::{
                        NodalTemperatures,
                        conduction::{
                            NodalForcesThermal, NodalStiffnessesThermal,
                            ThermalConductionFiniteElementBlock,
                        },
                    },
                },
                math::test::{TestError, assert_eq_from_fd},
            };
            mod finite_difference {
                use super::*;
                const MODEL: Fourier = Fourier {
                    thermal_conductivity: 1.0,
                };
                #[test]
                fn potential() -> Result<(), TestError> {
                    let constitutive_model = MODEL;
                    let block = Block::<Fourier, $element, G, M, N, P>::from((
                        constitutive_model,
                        get_connectivity(),
                        get_reference_coordinates_block(),
                    ));
                    let mut finite_difference = 0.0;
                    let nodal_forces_fd: NodalForcesThermal = (0..D)
                        .map(|node| {
                            let mut nodal_temperatures = NodalTemperatures::zero(D);
                            nodal_temperatures[node] += 0.5 * EPSILON;
                            finite_difference = block.potential(&nodal_temperatures)?;
                            nodal_temperatures[node] -= EPSILON;
                            finite_difference -= block.potential(&nodal_temperatures)?;
                            Ok(finite_difference / EPSILON)
                        })
                        .collect::<Result<_, FiniteElementBlockError>>()?;
                    assert_eq_from_fd(
                        &nodal_forces_fd,
                        &block.nodal_forces(&NodalTemperatures::zero(D))?,
                    )
                }
                #[test]
                fn nodal_forces() -> Result<(), TestError> {
                    let constitutive_model = MODEL;
                    let block = Block::<Fourier, $element, G, M, N, P>::from((
                        constitutive_model,
                        get_connectivity(),
                        get_reference_coordinates_block(),
                    ));
                    let mut finite_difference = 0.0;
                    let nodal_stiffnesses_fd: NodalStiffnessesThermal = (0..D)
                        .map(|node_a| {
                            (0..D)
                                .map(|node_b| {
                                    let mut nodal_temperatures = NodalTemperatures::zero(D);
                                    nodal_temperatures[node_b] += 0.5 * EPSILON;
                                    finite_difference =
                                        block.nodal_forces(&nodal_temperatures)?[node_a];
                                    nodal_temperatures[node_b] -= EPSILON;
                                    finite_difference -=
                                        block.nodal_forces(&nodal_temperatures)?[node_a];
                                    Ok(finite_difference / EPSILON)
                                })
                                .collect()
                        })
                        .collect::<Result<_, FiniteElementBlockError>>()?;
                    assert_eq_from_fd(
                        &nodal_stiffnesses_fd,
                        &block.nodal_stiffnesses(&NodalTemperatures::zero(D))?,
                    )
                }
            }
        }
    };
}
pub(crate) use test_thermal;
