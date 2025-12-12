macro_rules! test_thermal {
    ($element: ident) => {
        mod thermal_element {
            use super::{N, element};
            use crate::{
                EPSILON,
                constitutive::thermal::conduction::Fourier,
                fem::{
                    NodalForcesThermal, NodalStiffnessesThermal, NodalTemperatures,
                    block::element::{
                        FiniteElementError, thermal::conduction::ThermalConductionFiniteElement,
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
                    let element = element();
                    let temperature =
                        NodalTemperatures::from([0.62895714, 0.73331084, 0.3058115, 0.08179408]);
                    let mut finite_difference = 0.0;
                    let nodal_forces_fd: NodalForcesThermal<N> = (0..N)
                        .map(|node| {
                            let mut nodal_temperatures = temperature.clone();
                            nodal_temperatures[node] += 0.5 * EPSILON;
                            finite_difference =
                                element.potential(&constitutive_model, &nodal_temperatures)?;
                            nodal_temperatures[node] -= EPSILON;
                            finite_difference -=
                                element.potential(&constitutive_model, &nodal_temperatures)?;
                            Ok(finite_difference / EPSILON)
                        })
                        .collect::<Result<_, FiniteElementError>>()?;
                    assert_eq_from_fd(
                        &nodal_forces_fd,
                        &element.nodal_forces(
                            &constitutive_model,
                            &NodalTemperatures::from(temperature),
                        )?,
                    )
                }
                #[test]
                fn nodal_forces() -> Result<(), TestError> {
                    let constitutive_model = MODEL;
                    let element = element();
                    let temperature =
                        NodalTemperatures::from([0.62895714, 0.73331084, 0.3058115, 0.08179408]);
                    let mut finite_difference = 0.0;
                    let nodal_stiffnesses_fd: NodalStiffnessesThermal<N> = (0..N)
                        .map(|node_a| {
                            (0..N)
                                .map(|node_b| {
                                    let mut nodal_temperatures = temperature.clone();
                                    nodal_temperatures[node_b] += 0.5 * EPSILON;
                                    finite_difference = element
                                        .nodal_forces(&constitutive_model, &nodal_temperatures)?
                                        [node_a];
                                    nodal_temperatures[node_b] -= EPSILON;
                                    finite_difference -= element
                                        .nodal_forces(&constitutive_model, &nodal_temperatures)?
                                        [node_a];
                                    Ok(finite_difference / EPSILON)
                                })
                                .collect()
                        })
                        .collect::<Result<_, FiniteElementError>>()?;
                    assert_eq_from_fd(
                        &nodal_stiffnesses_fd,
                        &element.nodal_stiffnesses(
                            &constitutive_model,
                            &NodalTemperatures::from(temperature),
                        )?,
                    )
                }
            }
        }
    };
}
pub(crate) use test_thermal;
