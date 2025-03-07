macro_rules! test_linear_element {
    ($element: ident) => {
        crate::fem::block::element::test::setup_for_elements!($element);
        crate::fem::block::element::linear::test::test_linear_element_inner!($element);
        crate::fem::block::element::linear::test::test_linear_element_with_constitutive_model_size!(
            $element
        );
    };
}
pub(crate) use test_linear_element;

// should really consolidate this with similar tests in composite/test.rs

macro_rules! test_linear_element_inner {
    ($element: ident) => {
        use crate::math::TensorArray;
        crate::fem::block::element::test::test_finite_element!($element);
        mod linear_element {
            use super::*;
            use crate::{
                fem::block::element::linear::test::test_linear_element_with_constitutive_model,
                math::test::{assert_eq, assert_eq_within_tols, TestError},
            };
            mod elastic {
                use super::*;
                use crate::constitutive::solid::elastic::{
                    test::ALMANSIHAMELPARAMETERS, AlmansiHamel,
                };
                mod almansi_hamel {
                    use super::*;
                    test_linear_element_with_constitutive_model!(
                        $element,
                        AlmansiHamel,
                        ALMANSIHAMELPARAMETERS
                    );
                }
            }
            mod hyperelastic {
                use super::*;
                use crate::constitutive::solid::hyperelastic::{
                    test::{
                        ARRUDABOYCEPARAMETERS, FUNGPARAMETERS, GENTPARAMETERS,
                        MOONEYRIVLINPARAMETERS, NEOHOOKEANPARAMETERS,
                        SAINTVENANTKIRCHOFFPARAMETERS, YEOHPARAMETERS,
                    },
                    ArrudaBoyce, Fung, Gent, MooneyRivlin, NeoHookean, SaintVenantKirchhoff, Yeoh,
                };
                mod arruda_boyce {
                    use super::*;
                    test_linear_element_with_constitutive_model!(
                        $element,
                        ArrudaBoyce,
                        ARRUDABOYCEPARAMETERS
                    );
                }
                mod fung {
                    use super::*;
                    test_linear_element_with_constitutive_model!($element, Fung, FUNGPARAMETERS);
                }
                mod gent {
                    use super::*;
                    test_linear_element_with_constitutive_model!($element, Gent, GENTPARAMETERS);
                }
                mod mooney_rivlin {
                    use super::*;
                    test_linear_element_with_constitutive_model!(
                        $element,
                        MooneyRivlin,
                        MOONEYRIVLINPARAMETERS
                    );
                }
                mod neo_hookean {
                    use super::*;
                    test_linear_element_with_constitutive_model!(
                        $element,
                        NeoHookean,
                        NEOHOOKEANPARAMETERS
                    );
                }
                mod saint_venant_kirchhoff {
                    use super::*;
                    test_linear_element_with_constitutive_model!(
                        $element,
                        SaintVenantKirchhoff,
                        SAINTVENANTKIRCHOFFPARAMETERS
                    );
                }
                mod yeoh {
                    use super::*;
                    test_linear_element_with_constitutive_model!($element, Yeoh, YEOHPARAMETERS);
                }
            }
            mod elastic_hyperviscous {
                use super::*;
                use crate::constitutive::solid::elastic_hyperviscous::{
                    test::ALMANSIHAMELPARAMETERS, AlmansiHamel,
                };
                mod almansi_hamel {
                    use super::*;
                    test_linear_element_with_constitutive_model!(
                        $element,
                        AlmansiHamel,
                        ALMANSIHAMELPARAMETERS
                    );
                }
            }
            mod hyperviscoelastic {
                use super::*;
                use crate::constitutive::solid::hyperviscoelastic::{
                    test::SAINTVENANTKIRCHOFFPARAMETERS, SaintVenantKirchhoff,
                };
                mod saint_venant_kirchhoff {
                    use super::*;
                    test_linear_element_with_constitutive_model!(
                        $element,
                        SaintVenantKirchhoff,
                        SAINTVENANTKIRCHOFFPARAMETERS
                    );
                }
            }
        }
    };
}
pub(crate) use test_linear_element_inner;

macro_rules! test_linear_element_with_constitutive_model_size {
    ($element: ident) => {
        macro_rules! test_linear_element_with_constitutive_model_size_inner {
            ($elementt: ident, $constitutive_model: ident) => {
                #[test]
                fn size() {
                    assert_eq!(
                        std::mem::size_of::<$elementt::<$constitutive_model>>(),
                        std::mem::size_of::<$constitutive_model>()
                            + std::mem::size_of::<GradientVectors<G, N>>()
                            + std::mem::size_of::<Scalar>()
                    )
                }
            };
        }
        mod elastic {
            use super::*;
            use crate::constitutive::solid::elastic::AlmansiHamel;
            mod almansi_hamel {
                use super::*;
                test_linear_element_with_constitutive_model_size_inner!($element, AlmansiHamel);
            }
        }
        mod hyperelastic {
            use super::*;
            use crate::constitutive::solid::hyperelastic::{
                ArrudaBoyce, Fung, Gent, MooneyRivlin, NeoHookean, SaintVenantKirchhoff, Yeoh,
            };
            mod arruda_boyce {
                use super::*;
                test_linear_element_with_constitutive_model_size_inner!($element, ArrudaBoyce);
            }
            mod fung {
                use super::*;
                test_linear_element_with_constitutive_model_size_inner!($element, Fung);
            }
            mod gent {
                use super::*;
                test_linear_element_with_constitutive_model_size_inner!($element, Gent);
            }
            mod mooney_rivlin {
                use super::*;
                test_linear_element_with_constitutive_model_size_inner!($element, MooneyRivlin);
            }
            mod neo_hookean {
                use super::*;
                test_linear_element_with_constitutive_model_size_inner!($element, NeoHookean);
            }
            mod saint_venant_kirchhoff {
                use super::*;
                test_linear_element_with_constitutive_model_size_inner!(
                    $element,
                    SaintVenantKirchhoff
                );
            }
            mod yeoh {
                use super::*;
                test_linear_element_with_constitutive_model_size_inner!($element, Yeoh);
            }
        }
        mod elastic_hyperviscous {
            use super::*;
            use crate::constitutive::solid::elastic_hyperviscous::AlmansiHamel;
            mod almansi_hamel {
                use super::*;
                test_linear_element_with_constitutive_model_size_inner!($element, AlmansiHamel);
            }
        }
        mod hyperviscoelastic {
            use super::*;
            use crate::constitutive::solid::hyperviscoelastic::SaintVenantKirchhoff;
            mod saint_venant_kirchhoff {
                use super::*;
                test_linear_element_with_constitutive_model_size_inner!(
                    $element,
                    SaintVenantKirchhoff
                );
            }
        }
    };
}
pub(crate) use test_linear_element_with_constitutive_model_size;

macro_rules! setup_for_test_linear_element_with_constitutive_model {
    ($element: ident, $constitutive_model: ident, $constitutive_model_parameters: ident) => {
        fn get_element<'a>() -> $element<$constitutive_model<'a>> {
            $element::new($constitutive_model_parameters, get_reference_coordinates())
        }
        fn get_element_transformed<'a>() -> $element<$constitutive_model<'a>> {
            $element::<$constitutive_model>::new(
                $constitutive_model_parameters,
                get_reference_coordinates_transformed(),
            )
        }
    };
}
pub(crate) use setup_for_test_linear_element_with_constitutive_model;

macro_rules! test_linear_element_with_constitutive_model {
    ($element: ident, $constitutive_model: ident, $constitutive_model_parameters: ident) => {
        use crate::math::TensorArray;
        setup_for_test_linear_element_with_constitutive_model!(
            $element,
            $constitutive_model,
            $constitutive_model_parameters
        );
        mod deformation_gradient {
            fn get_deformation_gradients() -> DeformationGradients<G> {
                (0..G).map(|_| get_deformation_gradient()).collect()
            }
            use super::*;
            mod deformed {
                use super::*;
                #[test]
                fn calculate() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_element().deformation_gradients(&get_coordinates()),
                        &get_deformation_gradients(),
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    get_element()
                        .deformation_gradients(&get_coordinates())
                        .iter()
                        .zip(
                            get_element_transformed()
                                .deformation_gradients(&get_coordinates_transformed())
                                .iter(),
                        )
                        .try_for_each(|(deformation_gradient, deformation_gradient_transformed)| {
                            assert_eq_within_tols(
                                deformation_gradient,
                                &(get_rotation_current_configuration().transpose()
                                    * deformation_gradient_transformed
                                    * get_rotation_reference_configuration()),
                            )
                        })
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn calculate() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_element().deformation_gradients(&get_reference_coordinates().into()),
                        &DeformationGradients::identity(),
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_element_transformed()
                            .deformation_gradients(&get_reference_coordinates_transformed().into()),
                        &DeformationGradients::identity(),
                    )
                }
            }
        }
        mod deformation_gradient_rate {
            use super::*;
            mod deformed {
                fn get_deformation_gradient_rates() -> DeformationGradientRates<G> {
                    (0..G).map(|_| get_deformation_gradient_rate()).collect()
                }
                use super::*;
                #[test]
                fn calculate() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_element()
                            .deformation_gradient_rates(&get_coordinates(), &get_velocities()),
                        &get_deformation_gradient_rates(),
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    get_element()
                        .deformation_gradients(&get_coordinates())
                        .iter()
                        .zip(
                            get_element()
                                .deformation_gradient_rates(&get_coordinates(), &get_velocities())
                                .iter()
                                .zip(
                                    get_element_transformed()
                                        .deformation_gradient_rates(
                                            &get_coordinates_transformed(),
                                            &get_velocities_transformed(),
                                        )
                                        .iter(),
                                ),
                        )
                        .try_for_each(
                            |(
                                deformation_gradient,
                                (deformation_gradient_rate, deformation_gradient_rate_transformed),
                            )| {
                                assert_eq_within_tols(
                                    deformation_gradient_rate,
                                    &(get_rotation_current_configuration().transpose() * (
                                        deformation_gradient_rate_transformed
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
                fn calculate() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_element().deformation_gradient_rates(
                            &get_reference_coordinates().into(),
                            &NodalVelocities::zero().into(),
                        ),
                        &DeformationGradientRates::zero(),
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_element_transformed().deformation_gradient_rates(
                            &get_reference_coordinates_transformed().into(),
                            &NodalVelocities::zero().into(),
                        ),
                        &DeformationGradientRates::zero(),
                    )
                }
            }
            mod standard_gradient_operator {
                use super::*;
                #[test]
                fn partition_of_unity() -> Result<(), TestError> {
                    let mut sum = [0.0_f64; 3];
                    $element::<$constitutive_model>::standard_gradient_operator()
                        .iter()
                        .for_each(|row| {
                            row.iter()
                                .zip(sum.iter_mut())
                                .for_each(|(entry, sum_i)| *sum_i += entry)
                        });
                    sum.iter().try_for_each(|sum_i| assert_eq(sum_i, &0.0))
                }
            }
        }
    };
}
pub(crate) use test_linear_element_with_constitutive_model;
