macro_rules! test_dielectric {
    ($element: ident) => {
        mod dielectric_element {
            use super::{N, element};
            use crate::{
                EPSILON,
                constitutive::electrical::dielectric::Linear,
                fem::block::element::{
                    FiniteElementError,
                    electrical::{
                        ElectricalFiniteElement, ElementNodalPotentials,
                        dielectric::{
                            DielectricFiniteElement, ElementNodalForcesElectrical,
                            ElementNodalStiffnessesElectrical,
                        },
                    },
                },
                math::{Tensor, assert::AssertionError},
            };
            mod finite_difference {
                use super::*;
                use $crate::units::Permittivity;
                const MODEL: Linear = Linear {
                    permittivity: Permittivity::farads_per_meter(2.5),
                };
                fn potentials() -> ElementNodalPotentials<N> {
                    ElementNodalPotentials::from(
                        [0.62895714, 0.73331084, 0.3058115, 0.08179408]
                            .map($crate::math::Quantity::new),
                    )
                }
                #[test]
                fn potential() -> Result<(), AssertionError> {
                    let constitutive_model = MODEL;
                    let element = element();
                    let mut finite_difference = $crate::math::Quantity::default();
                    let nodal_forces_fd: ElementNodalForcesElectrical<N> = (0..N)
                        .map(|node| {
                            let mut nodal_potentials = potentials();
                            nodal_potentials[node] +=
                                $crate::math::assert::perturbation(0.5 * EPSILON);
                            finite_difference =
                                element.potential(&constitutive_model, &nodal_potentials)?;
                            nodal_potentials[node] -= $crate::math::assert::perturbation(EPSILON);
                            finite_difference -=
                                element.potential(&constitutive_model, &nodal_potentials)?;
                            Ok(finite_difference
                                / $crate::math::Quantity::<$crate::units::ElectricPotential>::new(
                                    EPSILON,
                                ))
                        })
                        .collect::<Result<_, FiniteElementError>>()?;
                    $crate::math::assert::Assert::default().eq_within_fd_tol(
                        &nodal_forces_fd,
                        &element.nodal_forces(&constitutive_model, &potentials())?,
                    )
                }
                #[test]
                fn nodal_forces() -> Result<(), AssertionError> {
                    let constitutive_model = MODEL;
                    let element = element();
                    let mut finite_difference = $crate::math::Quantity::default();
                    let nodal_stiffnesses_fd: ElementNodalStiffnessesElectrical<N> =
                        (0..N)
                            .map(|node_a| {
                                (0..N)
                                    .map(|node_b| {
                                        let mut nodal_potentials = potentials();
                                        nodal_potentials[node_b] +=
                                            $crate::math::assert::perturbation(0.5 * EPSILON);
                                        finite_difference = element
                                            .nodal_forces(&constitutive_model, &nodal_potentials)?
                                            [node_a];
                                        nodal_potentials[node_b] -=
                                            $crate::math::assert::perturbation(EPSILON);
                                        finite_difference -= element
                                            .nodal_forces(&constitutive_model, &nodal_potentials)?
                                            [node_a];
                                        Ok(finite_difference
                                            / $crate::math::Quantity::<
                                                $crate::units::ElectricPotential,
                                            >::new(EPSILON))
                                    })
                                    .collect()
                            })
                            .collect::<Result<_, FiniteElementError>>()?;
                    $crate::math::assert::Assert::default().eq_within_fd_tol(
                        &nodal_stiffnesses_fd,
                        &element.nodal_stiffnesses(&constitutive_model, &potentials())?,
                    )
                }
                #[test]
                fn a_uniform_potential_has_no_field_and_no_charge() -> Result<(), AssertionError> {
                    let constitutive_model = MODEL;
                    let element = element();
                    let potentials =
                        ElementNodalPotentials::from([7.3; N].map($crate::math::Quantity::new));
                    element
                        .nodal_forces(&constitutive_model, &potentials)?
                        .iter()
                        .try_for_each($crate::math::assert::Assert::zero)?;
                    element
                        .electric_fields(&potentials)
                        .iter()
                        .try_for_each($crate::math::assert::Assert::zero)
                }
                #[test]
                fn the_stiffness_is_symmetric_and_annihilates_a_uniform_potential()
                -> Result<(), AssertionError> {
                    let constitutive_model = MODEL;
                    let element = element();
                    let stiffnesses =
                        element.nodal_stiffnesses(&constitutive_model, &potentials())?;
                    for a in 0..N {
                        let row: $crate::math::Quantity<$crate::units::Capacitance> =
                            stiffnesses[a].iter().copied().sum();
                        $crate::math::assert::Assert::default().zero_within_tols(&row)?;
                        for b in 0..N {
                            $crate::math::assert::Assert::default()
                                .eq_within_tols(&stiffnesses[a][b], &stiffnesses[b][a])?
                        }
                    }
                    Ok(())
                }
            }
        }
    };
}
pub(crate) use test_dielectric;
