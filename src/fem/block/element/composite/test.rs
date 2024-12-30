macro_rules! test_composite_element {
    ($element: ident) => {
        crate::fem::block::element::test::setup_for_composite_elements!($element);
        crate::fem::block::element::composite::test::test_composite_element_inner!($element);
    };
}
pub(crate) use test_composite_element;

macro_rules! test_composite_element_inner
{
    ($element: ident) =>
    {
        crate::fem::block::element::test::test_finite_element!($element);
        mod composite
        {
            use crate::
            {
                fem::block::element::composite::test::test_composite_element_with_constitutive_model,
                math::test::{
                    assert_eq, assert_eq_within_tols, TestError,
                },
            };
            use super::*;
            mod elastic
            {
                use crate::
                {
                    constitutive::solid::elastic::
                    {
                        AlmansiHamel,
                        test::ALMANSIHAMELPARAMETERS
                    }
                };
                use super::*;
                mod almansi_hamel
                {
                    use super::*;
                    test_composite_element_with_constitutive_model!($element, AlmansiHamel, ALMANSIHAMELPARAMETERS);
                }
            }
            mod hyperelastic
            {
                use crate::
                {
                    constitutive::solid::hyperelastic::
                    {
                        ArrudaBoyce,
                        Fung,
                        Gent,
                        MooneyRivlin,
                        NeoHookean,
                        SaintVenantKirchoff,
                        Yeoh,
                        test::
                        {
                            ARRUDABOYCEPARAMETERS,
                            FUNGPARAMETERS,
                            GENTPARAMETERS,
                            MOONEYRIVLINPARAMETERS,
                            NEOHOOKEANPARAMETERS,
                            SAINTVENANTKIRCHOFFPARAMETERS,
                            YEOHPARAMETERS
                        }
                    }
                };
                use super::*;
                mod arruda_boyce
                {
                    use super::*;
                    test_composite_element_with_constitutive_model!($element, ArrudaBoyce, ARRUDABOYCEPARAMETERS);
                }
                mod fung
                {
                    use super::*;
                    test_composite_element_with_constitutive_model!($element, Fung, FUNGPARAMETERS);
                }
                mod gent
                {
                    use super::*;
                    test_composite_element_with_constitutive_model!($element, Gent, GENTPARAMETERS);
                }
                mod mooney_rivlin
                {
                    use super::*;
                    test_composite_element_with_constitutive_model!($element, MooneyRivlin, MOONEYRIVLINPARAMETERS);
                }
                mod neo_hookean
                {
                    use super::*;
                    test_composite_element_with_constitutive_model!($element, NeoHookean, NEOHOOKEANPARAMETERS);
                }
                mod saint_venant_kirchoff
                {
                    use super::*;
                    test_composite_element_with_constitutive_model!($element, SaintVenantKirchoff, SAINTVENANTKIRCHOFFPARAMETERS);
                }
                mod yeoh
                {
                    use super::*;
                    test_composite_element_with_constitutive_model!($element, Yeoh, YEOHPARAMETERS);
                }
            }
            mod elastic_hyperviscous
            {
                use crate::
                {
                    constitutive::solid::elastic_hyperviscous::
                    {
                        AlmansiHamel,
                        test::ALMANSIHAMELPARAMETERS
                    }
                };
                use super::*;
                mod almansi_hamel
                {
                    use super::*;
                    test_composite_element_with_constitutive_model!($element, AlmansiHamel, ALMANSIHAMELPARAMETERS);
                }
            }
            mod hyperviscoelastic
            {
                use crate::
                {
                    constitutive::solid::hyperviscoelastic::
                    {
                        SaintVenantKirchoff,
                        test::SAINTVENANTKIRCHOFFPARAMETERS
                    }
                };
                use super::*;
                mod saint_venant_kirchoff
                {
                    use super::*;
                    test_composite_element_with_constitutive_model!($element, SaintVenantKirchoff, SAINTVENANTKIRCHOFFPARAMETERS);
                }
            }
        }
    }
}
pub(crate) use test_composite_element_inner;

macro_rules! setup_for_test_composite_element_with_constitutive_model {
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
        #[test]
        fn size() {
            assert_eq!(
                std::mem::size_of::<$element::<$constitutive_model>>(),
                std::mem::size_of::<[$constitutive_model; G]>()
                    + std::mem::size_of::<ProjectedGradientVectors<G, N>>()
                    + std::mem::size_of::<Scalars<G>>()
            )
        }
    };
}
pub(crate) use setup_for_test_composite_element_with_constitutive_model;

macro_rules! test_composite_element_with_constitutive_model
{
    ($element: ident, $constitutive_model: ident, $constitutive_model_parameters: ident) =>
    {
        setup_for_test_composite_element_with_constitutive_model!($element, $constitutive_model, $constitutive_model_parameters);
        mod deformation_gradients
        {
            fn get_deformation_gradients() -> DeformationGradients<G> {
                (0..G).map(|_|
                    get_deformation_gradient()
                ).collect()
            }
            use super::*;
            mod deformed
            {
                use super::*;
                #[test]
                fn calculate() -> Result<(), TestError>
                {
                    assert_eq_within_tols(
                        &get_element().calculate_deformation_gradients(
                            &get_coordinates()
                        ),
                        &get_deformation_gradients()
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError>
                {
                    get_element().calculate_deformation_gradients(
                        &get_coordinates()
                    ).iter().zip((
                        get_element_transformed().calculate_deformation_gradients(
                            &get_coordinates_transformed()
                    )).iter())
                    .try_for_each(|(deformation_gradient, res_deformation_gradient)|
                        assert_eq_within_tols(
                            deformation_gradient,
                            &(get_rotation_current_configuration().transpose() * res_deformation_gradient * get_rotation_reference_configuration())
                        )
                    )
                }
            }
            mod undeformed
            {
                use super::*;
                #[test]
                fn calculate() -> Result<(), TestError>
                {
                    assert_eq_within_tols(
                        &get_element().calculate_deformation_gradients(
                            &get_reference_coordinates().into()
                        ),
                        &DeformationGradients::identity()
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError>
                {
                    assert_eq_within_tols(
                        &get_element_transformed().calculate_deformation_gradients(
                            &get_reference_coordinates_transformed().into()
                        ),
                        &DeformationGradients::identity()
                    )
                }
            }
        }
        mod deformation_gradient_rates
        {
            fn get_deformation_gradient_rates() -> DeformationGradientRates<G> {
                (0..G).map(|_|
                    get_deformation_gradient_rate()
                ).collect()
            }
            use super::*;
            mod deformed
            {
                use super::*;
                #[test]
                fn calculate() -> Result<(), TestError>
                {
                    assert_eq_within_tols(
                        &get_element().calculate_deformation_gradient_rates(
                            &get_coordinates(), &get_velocities()
                        ),
                        &get_deformation_gradient_rates()
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError>
                {
                    get_element().calculate_deformation_gradients(
                        &get_coordinates()
                    ).iter().zip((
                        get_element().calculate_deformation_gradient_rates(
                            &get_coordinates(), &get_velocities()
                        )
                    ).iter().zip((
                        get_element_transformed().calculate_deformation_gradient_rates(
                            &get_coordinates_transformed(), &get_velocities_transformed()
                        )
                    ).iter()))
                    .try_for_each(|(deformation_gradient, (deformation_gradient_rate, res_deformation_gradient_rate))|
                        assert_eq_within_tols(
                            deformation_gradient_rate,
                            &(
                                get_rotation_current_configuration().transpose() * (
                                    res_deformation_gradient_rate * get_rotation_reference_configuration()
                                    - get_rotation_rate_current_configuration() * deformation_gradient
                                )
                            )
                        )
                    )
                }
            }
            mod undeformed
            {
                use super::*;
                #[test]
                fn calculate() -> Result<(), TestError>
                {
                    assert_eq_within_tols(
                        &get_element_transformed().calculate_deformation_gradient_rates(
                            &get_reference_coordinates().into(),
                            &NodalVelocities::zero().into()
                        ),
                        &DeformationGradientRates::zero()
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError>
                {
                    assert_eq_within_tols(
                        &get_element_transformed().calculate_deformation_gradient_rates(
                            &get_reference_coordinates_transformed().into(),
                            &NodalVelocities::zero().into()
                        ),
                        &DeformationGradientRates::zero()
                    )
                }
            }
        }
        mod partition_of_unity
        {
            use super::*;
            #[test]
            fn shape_functions() -> Result<(), TestError>
            {
                $element::<$constitutive_model>::calculate_shape_functions_at_integration_points().iter()
                .try_for_each(|shape_functions|
                    assert_eq(&shape_functions.iter().sum(), &1.0)
                )
            }
            #[test]
            fn standard_gradient_operators() -> Result<(), TestError>
            {
                let mut sum = [0.0_f64; 3];
                $element::<$constitutive_model>::calculate_standard_gradient_operators().iter()
                .try_for_each(|standard_gradient_operator|{
                    standard_gradient_operator.iter()
                    .for_each(|row|
                        row.iter()
                        .zip(sum.iter_mut())
                        .for_each(|(entry, sum_i)|
                            *sum_i += entry
                        )
                    );
                    sum.iter()
                    .try_for_each(|sum_i|
                        assert_eq_within_tols(sum_i, &0.0)
                    )
                })
            }
        }
        #[test]
        fn normalized_projection_matrix() -> Result<(), TestError>
        {
            $element::<$constitutive_model>::calculate_shape_function_integrals_products()
            .iter().map(|dummy| dummy * 1.0).sum::<TensorRank2<Q, 9, 9>>().iter()
            .zip($element::<$constitutive_model>::calculate_inverse_normalized_projection_matrix()
            .inverse().iter())
            .try_for_each(|(sum_i, projection_matrix_i)|
                sum_i.iter()
                .zip(projection_matrix_i.iter())
                .try_for_each(|(sum_ij, projection_matrix_ij)|
                    assert_eq_within_tols(sum_ij, projection_matrix_ij)
                )
            )
        }
    }
}
pub(crate) use test_composite_element_with_constitutive_model;
