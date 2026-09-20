#![cfg_attr(
    not(any(feature = "fem", feature = "vem")),
    allow(unused_macros, unused_imports)
)]

macro_rules! test_solid_deformation_gradient {
    () => {
        mod solid {
            use super::*;
            fn deformation_gradients() -> DeformationGradientList {
                (0..number_of_gradients())
                    .map(|_| get_deformation_gradient())
                    .collect()
            }
            fn deformation_gradient_rates() -> DeformationGradientRateList {
                (0..number_of_gradients())
                    .map(|_| get_deformation_gradient_rate())
                    .collect()
            }
            mod deformation_gradient {
                use super::*;
                mod deformed {
                    use super::*;
                    #[test]
                    fn calculate() -> Result<(), AssertionError> {
                        $crate::math::assert::Assert::default().eq_within_tols(
                            &element().deformation_gradients(&coordinates()),
                            &deformation_gradients(),
                        )
                    }
                    #[test]
                    fn objectivity() -> Result<(), AssertionError> {
                        element()
                            .deformation_gradients(&coordinates())
                            .iter()
                            .zip(
                                element_transformed()
                                    .deformation_gradients(&coordinates_transformed())
                                    .iter(),
                            )
                            .try_for_each(
                                |(deformation_gradient, deformation_gradient_transformed)| {
                                    $crate::math::assert::Assert::default().eq_within_tols(
                                        deformation_gradient,
                                        &(get_rotation_current_configuration().transpose()
                                            * deformation_gradient_transformed
                                            * get_rotation_reference_configuration()),
                                    )
                                },
                            )
                    }
                }
                mod undeformed {
                    use super::*;
                    #[test]
                    fn calculate() -> Result<(), AssertionError> {
                        $crate::math::assert::Assert::default().eq_within_tols(
                            &element().deformation_gradients(&reference_coordinates().into()),
                            &identity_deformation_gradients(),
                        )
                    }
                    #[test]
                    fn objectivity() -> Result<(), AssertionError> {
                        $crate::math::assert::Assert::default().eq_within_tols(
                            &element_transformed()
                                .deformation_gradients(&reference_coordinates_transformed().into()),
                            &identity_deformation_gradients(),
                        )
                    }
                }
            }
            mod deformation_gradient_rate {
                use super::*;
                mod deformed {
                    use super::*;
                    #[test]
                    fn calculate() -> Result<(), AssertionError> {
                        $crate::math::assert::Assert::default().eq_within_tols(
                            &element().deformation_gradient_rates(&coordinates(), &velocities()),
                            &deformation_gradient_rates(),
                        )
                    }
                    #[test]
                    fn objectivity() -> Result<(), AssertionError> {
                        element()
                            .deformation_gradients(&coordinates())
                            .iter()
                            .zip(
                                element()
                                    .deformation_gradient_rates(&coordinates(), &velocities())
                                    .iter()
                                    .zip(
                                        element_transformed()
                                            .deformation_gradient_rates(
                                                &coordinates_transformed(),
                                                &velocities_transformed(),
                                            )
                                            .iter(),
                                    ),
                            )
                            .try_for_each(
                                |(
                                    deformation_gradient,
                                    (
                                        deformation_gradient_rate,
                                        deformation_gradient_rate_transformed,
                                    ),
                                )| {
                                    $crate::math::assert::Assert::default().eq_within_tols(
                                        deformation_gradient_rate,
                                        &(get_rotation_current_configuration().transpose()
                                            * (deformation_gradient_rate_transformed
                                                * get_rotation_reference_configuration()
                                                - get_rotation_rate_current_configuration()
                                                    * deformation_gradient)),
                                    )
                                },
                            )
                    }
                }
                mod undeformed {
                    use super::*;
                    #[test]
                    fn calculate() -> Result<(), AssertionError> {
                        $crate::math::assert::Assert::default().eq_within_tols(
                            &element().deformation_gradient_rates(
                                &reference_coordinates().into(),
                                &zero_velocities().into(),
                            ),
                            &zero_deformation_gradient_rates(),
                        )
                    }
                    #[test]
                    fn objectivity() -> Result<(), AssertionError> {
                        $crate::math::assert::Assert::default().eq_within_tols(
                            &element_transformed().deformation_gradient_rates(
                                &reference_coordinates_transformed().into(),
                                &zero_velocities().into(),
                            ),
                            &zero_deformation_gradient_rates(),
                        )
                    }
                }
            }
        }
    };
}
pub(crate) use test_solid_deformation_gradient;
