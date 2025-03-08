macro_rules! test_linear_element {
    ($element: ident) => {
        crate::fem::block::element::test::setup_for_elements!($element);
        crate::fem::block::element::linear::test::test_linear_element_inner!($element);
    };
}
pub(crate) use test_linear_element;

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
            mod foo {
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
        }
    };
}
pub(crate) use test_linear_element_inner;

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
    };
}
pub(crate) use test_linear_element_with_constitutive_model;
