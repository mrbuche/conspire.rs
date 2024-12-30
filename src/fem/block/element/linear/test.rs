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

macro_rules! test_linear_element_inner {
    ($element: ident) => {
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
                    ArrudaBoyce, Fung, Gent, MooneyRivlin, NeoHookean, SaintVenantKirchoff, Yeoh,
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
                mod saint_venant_kirchoff {
                    use super::*;
                    test_linear_element_with_constitutive_model!(
                        $element,
                        SaintVenantKirchoff,
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
                    test::SAINTVENANTKIRCHOFFPARAMETERS, SaintVenantKirchoff,
                };
                mod saint_venant_kirchoff {
                    use super::*;
                    test_linear_element_with_constitutive_model!(
                        $element,
                        SaintVenantKirchoff,
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
                            + std::mem::size_of::<GradientVectors<N>>()
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
                ArrudaBoyce, Fung, Gent, MooneyRivlin, NeoHookean, SaintVenantKirchoff, Yeoh,
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
            mod saint_venant_kirchoff {
                use super::*;
                test_linear_element_with_constitutive_model_size_inner!(
                    $element,
                    SaintVenantKirchoff
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
            use crate::constitutive::solid::hyperviscoelastic::SaintVenantKirchoff;
            mod saint_venant_kirchoff {
                use super::*;
                test_linear_element_with_constitutive_model_size_inner!(
                    $element,
                    SaintVenantKirchoff
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
        setup_for_test_linear_element_with_constitutive_model!(
            $element,
            $constitutive_model,
            $constitutive_model_parameters
        );
        mod deformation_gradient {
            use super::*;
            mod deformed {
                use super::*;
                #[test]
                fn calculate() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_element().calculate_deformation_gradient(&get_coordinates()),
                        &get_deformation_gradient(),
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_element().calculate_deformation_gradient(&get_coordinates()),
                        &(get_rotation_current_configuration().transpose()
                            * get_element_transformed()
                                .calculate_deformation_gradient(&get_coordinates_transformed())
                            * get_rotation_reference_configuration()),
                    )
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn calculate() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_element()
                            .calculate_deformation_gradient(&get_reference_coordinates().into()),
                        &DeformationGradient::identity(),
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_element_transformed().calculate_deformation_gradient(
                            &get_reference_coordinates_transformed().into(),
                        ),
                        &DeformationGradient::identity(),
                    )
                }
            }
        }
        mod deformation_gradient_rate {
            use super::*;
            mod deformed {
                use super::*;
                #[test]
                fn calculate() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_element().calculate_deformation_gradient_rate(
                            &get_coordinates(),
                            &get_velocities(),
                        ),
                        &get_deformation_gradient_rate(),
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_element().calculate_deformation_gradient_rate(
                            &get_coordinates(),
                            &get_velocities(),
                        ),
                        &(get_rotation_current_configuration().transpose()
                            * (get_element_transformed().calculate_deformation_gradient_rate(
                                &get_coordinates_transformed(),
                                &get_velocities_transformed(),
                            ) * get_rotation_reference_configuration()
                                - get_rotation_rate_current_configuration()
                                    * get_element()
                                        .calculate_deformation_gradient(&get_coordinates()))),
                    )
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn calculate() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_element().calculate_deformation_gradient_rate(
                            &get_reference_coordinates().into(),
                            &NodalVelocities::zero().into(),
                        ),
                        &DeformationGradientRate::zero(),
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_element_transformed().calculate_deformation_gradient_rate(
                            &get_reference_coordinates_transformed().into(),
                            &NodalVelocities::zero().into(),
                        ),
                        &DeformationGradientRate::zero(),
                    )
                }
            }
        }
        mod standard_gradient_operator {
            use super::*;
            #[test]
            fn partition_of_unity() -> Result<(), TestError> {
                let mut sum = [0.0_f64; 3];
                $element::<$constitutive_model>::calculate_standard_gradient_operator()
                    .iter()
                    .for_each(|row| {
                        row.iter()
                            .zip(sum.iter_mut())
                            .for_each(|(entry, sum_i)| *sum_i += entry)
                    });
                sum.iter().try_for_each(|sum_i| assert_eq(sum_i, &0.0))
            }
        }
    };
}
pub(crate) use test_linear_element_with_constitutive_model;
