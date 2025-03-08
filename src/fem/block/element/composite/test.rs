macro_rules! test_composite_element {
    ($element: ident) => {
        crate::fem::block::element::test::setup_for_elements!($element);
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
            mod foo
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
    };
}
pub(crate) use setup_for_test_composite_element_with_constitutive_model;

macro_rules! test_composite_element_with_constitutive_model {
    ($element: ident, $constitutive_model: ident, $constitutive_model_parameters: ident) => {
        use crate::math::TensorArray;
        setup_for_test_composite_element_with_constitutive_model!(
            $element,
            $constitutive_model,
            $constitutive_model_parameters
        );
        mod partition_of_unity {
            use super::*;
            #[test]
            fn shape_functions() -> Result<(), TestError> {
                $element::<$constitutive_model>::shape_functions_at_integration_points()
                    .iter()
                    .try_for_each(|shape_functions| assert_eq(&shape_functions.iter().sum(), &1.0))
            }
            // #[test]
            // fn standard_gradient_operators() -> Result<(), TestError> {
            //     let mut sum = [0.0_f64; 3];
            //     $element::<$constitutive_model>::standard_gradient_operators()
            //         .iter()
            //         .try_for_each(|standard_gradient_operator| {
            //             standard_gradient_operator.iter().for_each(|row| {
            //                 row.iter()
            //                     .zip(sum.iter_mut())
            //                     .for_each(|(entry, sum_i)| *sum_i += entry)
            //             });
            //             sum.iter()
            //                 .try_for_each(|sum_i| assert_eq_within_tols(sum_i, &0.0))
            //         })
            // }
        }
        #[test]
        fn normalized_projection_matrix() -> Result<(), TestError> {
            $element::<$constitutive_model>::shape_function_integrals_products()
                .iter()
                .map(|dummy| dummy * 1.0)
                .sum::<TensorRank2<Q, 9, 9>>()
                .iter()
                .zip(
                    $element::<$constitutive_model>::inverse_normalized_projection_matrix()
                        .inverse()
                        .iter(),
                )
                .try_for_each(|(sum_i, projection_matrix_i)| {
                    sum_i.iter().zip(projection_matrix_i.iter()).try_for_each(
                        |(sum_ij, projection_matrix_ij)| {
                            assert_eq_within_tols(sum_ij, projection_matrix_ij)
                        },
                    )
                })
        }
    };
}
pub(crate) use test_composite_element_with_constitutive_model;
