macro_rules! test_thermal {
    ($element: ident) => {
        mod thermal_element {
            use super::{N, element};
            use crate::{
                EPSILON,
                constitutive::thermal::conduction::Fourier,
                fem::block::element::{
                    FiniteElementError,
                    thermal::{
                        ElementNodalTemperatures,
                        conduction::{
                            ElementNodalForcesThermal, ElementNodalStiffnessesThermal,
                            ThermalConductionFiniteElement,
                        },
                    },
                },
                math::assert::AssertionError,
            };
            mod finite_difference {
                use super::*;
                use $crate::units::PowerPerLengthTemperature;
                const MODEL: Fourier = Fourier {
                    thermal_conductivity: PowerPerLengthTemperature::watts_per_meter_kelvin(1.0),
                };
                #[test]
                fn potential() -> Result<(), AssertionError> {
                    let constitutive_model = MODEL;
                    let element = element();
                    let temperature = ElementNodalTemperatures::from(
                        [0.62895714, 0.73331084, 0.3058115, 0.08179408]
                            .map($crate::math::Quantity::new),
                    );
                    let mut finite_difference = $crate::math::Quantity::default();
                    let nodal_forces_fd: ElementNodalForcesThermal<N> = (0..N)
                        .map(|node| {
                            let mut nodal_temperatures = temperature.clone();
                            nodal_temperatures[node] +=
                                $crate::math::assert::perturbation(0.5 * EPSILON);
                            finite_difference =
                                element.potential(&constitutive_model, &nodal_temperatures)?;
                            nodal_temperatures[node] -= $crate::math::assert::perturbation(EPSILON);
                            finite_difference -=
                                element.potential(&constitutive_model, &nodal_temperatures)?;
                            // A potential per unit temperature is a power.
                            Ok(finite_difference
                                / $crate::math::Quantity::<$crate::units::Temperature>::new(
                                    EPSILON,
                                ))
                        })
                        .collect::<Result<_, FiniteElementError>>()?;
                    $crate::math::assert::Assert::default().eq_within_fd_tol(
                        &nodal_forces_fd,
                        &element.nodal_forces(
                            &constitutive_model,
                            &ElementNodalTemperatures::from(temperature),
                        )?,
                    )
                }
                #[test]
                fn nodal_forces() -> Result<(), AssertionError> {
                    let constitutive_model = MODEL;
                    let element = element();
                    let temperature = ElementNodalTemperatures::from(
                        [0.62895714, 0.73331084, 0.3058115, 0.08179408]
                            .map($crate::math::Quantity::new),
                    );
                    let mut finite_difference = $crate::math::Quantity::default();
                    let nodal_stiffnesses_fd: ElementNodalStiffnessesThermal<N> = (0..N)
                        .map(|node_a| {
                            (0..N)
                                .map(|node_b| {
                                    let mut nodal_temperatures = temperature.clone();
                                    nodal_temperatures[node_b] +=
                                        $crate::math::assert::perturbation(0.5 * EPSILON);
                                    finite_difference = element
                                        .nodal_forces(&constitutive_model, &nodal_temperatures)?
                                        [node_a];
                                    nodal_temperatures[node_b] -=
                                        $crate::math::assert::perturbation(EPSILON);
                                    finite_difference -= element
                                        .nodal_forces(&constitutive_model, &nodal_temperatures)?
                                        [node_a];
                                    // A power per unit temperature is a thermal stiffness.
                                    Ok(finite_difference
                                        / $crate::math::Quantity::<$crate::units::Temperature>::new(
                                            EPSILON,
                                        ))
                                })
                                .collect()
                        })
                        .collect::<Result<_, FiniteElementError>>()?;
                    $crate::math::assert::Assert::default().eq_within_fd_tol(
                        &nodal_stiffnesses_fd,
                        &element.nodal_stiffnesses(
                            &constitutive_model,
                            &ElementNodalTemperatures::from(temperature),
                        )?,
                    )
                }
            }
        }
    };
}
pub(crate) use test_thermal;
