use crate::mechanics::Scalar;

pub const BULK_MODULUS: Scalar = 13.0;
pub const SHEAR_MODULUS: Scalar = 3.0;

macro_rules! cauchy_stress_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.cauchy_stress($deformation_gradient)
    };
}
pub(crate) use cauchy_stress_from_deformation_gradient;

macro_rules! cauchy_stress_from_deformation_gradient_simple {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.cauchy_stress($deformation_gradient)
    };
}
pub(crate) use cauchy_stress_from_deformation_gradient_simple;

macro_rules! cauchy_stress_from_deformation_gradient_rotated {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.cauchy_stress($deformation_gradient)
    };
}
pub(crate) use cauchy_stress_from_deformation_gradient_rotated;

macro_rules! cauchy_tangent_stiffness_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.cauchy_tangent_stiffness($deformation_gradient)
    };
}
pub(crate) use cauchy_tangent_stiffness_from_deformation_gradient;

macro_rules! first_piola_kirchhoff_stress_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.first_piola_kirchhoff_stress($deformation_gradient)
    };
}
pub(crate) use first_piola_kirchhoff_stress_from_deformation_gradient;

macro_rules! first_piola_kirchhoff_stress_from_deformation_gradient_simple {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.first_piola_kirchhoff_stress($deformation_gradient)
    };
}
pub(crate) use first_piola_kirchhoff_stress_from_deformation_gradient_simple;

macro_rules! first_piola_kirchhoff_stress_from_deformation_gradient_rotated {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.first_piola_kirchhoff_stress($deformation_gradient)
    };
}
pub(crate) use first_piola_kirchhoff_stress_from_deformation_gradient_rotated;

macro_rules! first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.first_piola_kirchhoff_tangent_stiffness($deformation_gradient)
    };
}
pub(crate) use first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient;

macro_rules! first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient_simple {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.first_piola_kirchhoff_tangent_stiffness($deformation_gradient)
    };
}
pub(crate) use first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient_simple;

macro_rules! second_piola_kirchhoff_stress_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.second_piola_kirchhoff_stress($deformation_gradient)
    };
}
pub(crate) use second_piola_kirchhoff_stress_from_deformation_gradient;

macro_rules! second_piola_kirchhoff_stress_from_deformation_gradient_simple {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.second_piola_kirchhoff_stress($deformation_gradient)
    };
}
pub(crate) use second_piola_kirchhoff_stress_from_deformation_gradient_simple;

macro_rules! second_piola_kirchhoff_stress_from_deformation_gradient_rotated {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.second_piola_kirchhoff_stress($deformation_gradient)
    };
}
pub(crate) use second_piola_kirchhoff_stress_from_deformation_gradient_rotated;

macro_rules! second_piola_kirchhoff_tangent_stiffness_from_deformation_gradient {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.second_piola_kirchhoff_tangent_stiffness($deformation_gradient)
    };
}
pub(crate) use second_piola_kirchhoff_tangent_stiffness_from_deformation_gradient;

macro_rules! test_solid_constitutive_model {
    ($constitutive_model: expr) => {
        crate::constitutive::solid::elastic::test::test_solid_constitutive!($constitutive_model);
        crate::constitutive::solid::elastic::test::test_constructed_solid_constitutive_model!(
            $constitutive_model
        );
    };
}
pub(crate) use test_solid_constitutive_model;

macro_rules! test_constructed_solid_constitutive_model {
    ($constitutive_model: expr) => {
        crate::constitutive::solid::elastic::test::test_solid_constitutive_model_no_tangents!(
            $constitutive_model
        );
        crate::constitutive::solid::elastic::test::test_solid_constitutive_model_tangents!(
            $constitutive_model
        );
    };
}
pub(crate) use test_constructed_solid_constitutive_model;

macro_rules! test_solid_constitutive
{
    ($constitutive_model: expr) =>
    {
        #[test]
        fn bulk_modulus() -> Result<(), TestError>
        {
            use crate::EPSILON;
            let model = $constitutive_model;
            let deformation_gradient = DeformationGradient::identity()*(1.0 + EPSILON / 3.0);
            let first_piola_kirchhoff_stress = first_piola_kirchhoff_stress_from_deformation_gradient_simple!(&model, &deformation_gradient)?;
            assert!((3.0 * EPSILON * model.bulk_modulus() / first_piola_kirchhoff_stress.trace() - 1.0).abs() < EPSILON);
            Ok(())
        }
        #[test]
        fn shear_modulus() -> Result<(), TestError>
        {
            use crate::EPSILON;
            let model = $constitutive_model;
            let mut deformation_gradient = DeformationGradient::identity();
            deformation_gradient[0][1] = EPSILON;
            let first_piola_kirchhoff_stress = first_piola_kirchhoff_stress_from_deformation_gradient_simple!(&model, &deformation_gradient)?;
            assert!((EPSILON * model.shear_modulus() / first_piola_kirchhoff_stress[0][1] - 1.0).abs() < EPSILON);
            Ok(())
        }
    }
}
pub(crate) use test_solid_constitutive;

macro_rules! test_solid_constitutive_model_no_tangents {
    ($constitutive_model: expr) => {
        use crate::{
            EPSILON,
            math::{
                TensorArray,
                test::{TestError, assert_eq, assert_eq_within_tols},
            },
            mechanics::{
                CauchyStress, FirstPiolaKirchhoffStress, SecondPiolaKirchhoffStress,
                test::{
                    get_deformation_gradient, get_deformation_gradient_rotated,
                    get_rotation_current_configuration, get_rotation_reference_configuration,
                },
            },
        };
        mod cauchy_stress {
            use super::*;
            mod deformed {
                use super::*;
                #[test]
                #[should_panic(expected = "Invalid Jacobian")]
                fn invalid_jacobian() {
                    let _ = EPSILON;
                    let mut deformation_gradient = DeformationGradient::identity();
                    deformation_gradient[0][0] *= -1.0;
                    cauchy_stress_from_deformation_gradient!(
                        $constitutive_model,
                        &deformation_gradient
                    )
                    .unwrap();
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &cauchy_stress_from_deformation_gradient!(
                            &$constitutive_model,
                            &get_deformation_gradient()
                        )?,
                        &(get_rotation_current_configuration().transpose()
                            * cauchy_stress_from_deformation_gradient_rotated!(
                                &$constitutive_model,
                                &get_deformation_gradient_rotated()
                            )?
                            * get_rotation_current_configuration()),
                    )
                }
                #[test]
                fn symmetry() -> Result<(), TestError> {
                    let cauchy_stress = cauchy_stress_from_deformation_gradient!(
                        &$constitutive_model,
                        &get_deformation_gradient()
                    )?;
                    assert_eq_within_tols(&cauchy_stress, &cauchy_stress.transpose())
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn zero() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &cauchy_stress_from_deformation_gradient_simple!(
                            &$constitutive_model,
                            &DeformationGradient::identity()
                        )?,
                        &CauchyStress::zero(),
                    )
                }
            }
        }
        mod first_piola_kirchhoff_stress {
            use super::*;
            mod deformed {
                #[test]
                #[should_panic(expected = "Invalid Jacobian")]
                fn invalid_jacobian() {
                    let mut deformation_gradient = DeformationGradient::identity();
                    deformation_gradient[0][0] *= -1.0;
                    first_piola_kirchhoff_stress_from_deformation_gradient!(
                        $constitutive_model,
                        &deformation_gradient
                    )
                    .unwrap();
                }
                use super::*;
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &first_piola_kirchhoff_stress_from_deformation_gradient!(
                            &$constitutive_model,
                            &get_deformation_gradient()
                        )?,
                        &(get_rotation_current_configuration().transpose()
                            * first_piola_kirchhoff_stress_from_deformation_gradient_rotated!(
                                &$constitutive_model,
                                &get_deformation_gradient_rotated()
                            )?
                            * get_rotation_reference_configuration()),
                    )
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn zero() -> Result<(), TestError> {
                    assert_eq(
                        &first_piola_kirchhoff_stress_from_deformation_gradient_simple!(
                            &$constitutive_model,
                            &DeformationGradient::identity()
                        )?,
                        &FirstPiolaKirchhoffStress::zero(),
                    )
                }
            }
        }
        mod second_piola_kirchhoff_stress {
            use super::*;
            mod deformed {
                #[test]
                #[should_panic(expected = "Invalid Jacobian")]
                fn invalid_jacobian() {
                    let mut deformation_gradient = DeformationGradient::identity();
                    deformation_gradient[0][0] *= -1.0;
                    second_piola_kirchhoff_stress_from_deformation_gradient!(
                        $constitutive_model,
                        &deformation_gradient
                    )
                    .unwrap();
                }
                use super::*;
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &second_piola_kirchhoff_stress_from_deformation_gradient!(
                            &$constitutive_model,
                            &get_deformation_gradient()
                        )?,
                        &(get_rotation_reference_configuration().transpose()
                            * second_piola_kirchhoff_stress_from_deformation_gradient_rotated!(
                                &$constitutive_model,
                                &get_deformation_gradient_rotated()
                            )?
                            * get_rotation_reference_configuration()),
                    )
                }
                #[test]
                fn symmetry() -> Result<(), TestError> {
                    let second_piola_kirchhoff_stress = second_piola_kirchhoff_stress_from_deformation_gradient!(
                        &$constitutive_model,
                        &get_deformation_gradient()
                    )?;
                    assert_eq_within_tols(&second_piola_kirchhoff_stress, &second_piola_kirchhoff_stress.transpose())
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn zero() -> Result<(), TestError> {
                    assert_eq(
                        &second_piola_kirchhoff_stress_from_deformation_gradient_simple!(
                            &$constitutive_model,
                            &DeformationGradient::identity()
                        )?,
                        &SecondPiolaKirchhoffStress::zero(),
                    )
                }
            }
        }
    };
}
pub(crate) use test_solid_constitutive_model_no_tangents;

macro_rules! test_solid_constitutive_model_tangents
{
    ($constitutive_model: expr) =>
    {
        mod tangents
        {
            use crate::
            {
                math::{ContractAllIndicesWithFirstIndicesOf, test::assert_eq_from_fd, TensorArray},
                mechanics::{FirstPiolaKirchhoffTangentStiffness, SecondPiolaKirchhoffTangentStiffness, test::get_deformation_gradient_rotated_undeformed},
            };
            use super::*;
            fn cauchy_tangent_stiffness_from_finite_difference_of_cauchy_stress(is_deformed: bool) -> Result<CauchyTangentStiffness, TestError>
            {
                let model = $constitutive_model;
                let mut cauchy_tangent_stiffness = CauchyTangentStiffness::zero();
                for k in 0..3
                {
                    for l in 0..3
                    {
                        let mut deformation_gradient_plus =
                            if is_deformed
                            {
                                get_deformation_gradient()
                            }
                            else
                            {
                                DeformationGradient::identity()
                            };
                        deformation_gradient_plus[k][l] += 0.5*EPSILON;
                        let cauchy_stress_plus = cauchy_stress_from_deformation_gradient!(&model, &deformation_gradient_plus)?;
                        let mut deformation_gradient_minus =
                            if is_deformed
                            {
                                get_deformation_gradient()
                            }
                            else
                            {
                                DeformationGradient::identity()
                            };
                        deformation_gradient_minus[k][l] -= 0.5*EPSILON;
                        let cauchy_stress_minus = cauchy_stress_from_deformation_gradient!(&model, &deformation_gradient_minus)?;
                        for i in 0..3
                        {
                            for j in 0..3
                            {
                                cauchy_tangent_stiffness[i][j][k][l] = (cauchy_stress_plus[i][j] - cauchy_stress_minus[i][j])/EPSILON;
                            }
                        }
                    }
                }
                Ok(cauchy_tangent_stiffness)
            }
            fn first_piola_kirchhoff_tangent_stiffness_from_finite_difference_of_first_piola_kirchhoff_stress(is_deformed: bool) -> Result<FirstPiolaKirchhoffTangentStiffness, TestError>
            {
                let model = $constitutive_model;
                let mut first_piola_kirchhoff_tangent_stiffness = FirstPiolaKirchhoffTangentStiffness::zero();
                for k in 0..3
                {
                    for l in 0..3
                    {
                        let mut deformation_gradient_plus =
                            if is_deformed
                            {
                                get_deformation_gradient()
                            }
                            else
                            {
                                DeformationGradient::identity()
                            };
                        deformation_gradient_plus[k][l] += 0.5*EPSILON;
                        let first_piola_kirchhoff_stress_plus = first_piola_kirchhoff_stress_from_deformation_gradient!(&model, &deformation_gradient_plus)?;
                        let mut deformation_gradient_minus =
                            if is_deformed
                            {
                                get_deformation_gradient()
                            }
                            else
                            {
                                DeformationGradient::identity()
                            };
                        deformation_gradient_minus[k][l] -= 0.5*EPSILON;
                        let first_piola_kirchhoff_stress_minus = first_piola_kirchhoff_stress_from_deformation_gradient!(&model, &deformation_gradient_minus)?;
                        for i in 0..3
                        {
                            for j in 0..3
                            {
                                first_piola_kirchhoff_tangent_stiffness[i][j][k][l] = (first_piola_kirchhoff_stress_plus[i][j] - first_piola_kirchhoff_stress_minus[i][j])/EPSILON;
                            }
                        }
                    }
                }
                Ok(first_piola_kirchhoff_tangent_stiffness)
            }
            fn second_piola_kirchhoff_tangent_stiffness_from_finite_difference_of_second_piola_kirchhoff_stress(is_deformed: bool) -> Result<SecondPiolaKirchhoffTangentStiffness, TestError>
            {
                let mut second_piola_kirchhoff_tangent_stiffness = SecondPiolaKirchhoffTangentStiffness::zero();
                for k in 0..3
                {
                    for l in 0..3
                    {
                        let mut deformation_gradient_plus =
                            if is_deformed
                            {
                                get_deformation_gradient()
                            }
                            else
                            {
                                DeformationGradient::identity()
                            };
                        deformation_gradient_plus[k][l] += 0.5*EPSILON;
                        let second_piola_kirchhoff_stress_plus =
                        second_piola_kirchhoff_stress_from_deformation_gradient!(
                            $constitutive_model, &deformation_gradient_plus
                        )?;
                        let mut deformation_gradient_minus =
                            if is_deformed
                            {
                                get_deformation_gradient()
                            }
                            else
                            {
                                DeformationGradient::identity()
                            };
                        deformation_gradient_minus[k][l] -= 0.5*EPSILON;
                        let second_piola_kirchhoff_stress_minus =
                        second_piola_kirchhoff_stress_from_deformation_gradient!(
                            $constitutive_model, &deformation_gradient_minus
                        )?;
                        for i in 0..3
                        {
                            for j in 0..3
                            {
                                second_piola_kirchhoff_tangent_stiffness[i][j][k][l] = (
                                    second_piola_kirchhoff_stress_plus[i][j] - second_piola_kirchhoff_stress_minus[i][j]
                                )/EPSILON;
                            }
                        }
                    }
                }
                Ok(second_piola_kirchhoff_tangent_stiffness)
            }
            mod cauchy_stress
            {
                use super::*;
                mod deformed
                {
                    use super::*;
                    #[test]
                    fn finite_difference() -> Result<(), TestError>
                    {
                        assert_eq_from_fd(
                            &cauchy_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &get_deformation_gradient()
                            )?,
                            &cauchy_tangent_stiffness_from_finite_difference_of_cauchy_stress(true)?,
                        )
                    }
                }
                mod undeformed
                {
                    use super::*;
                    #[test]
                    fn finite_difference() -> Result<(), TestError>
                    {
                        assert_eq_from_fd(
                            &cauchy_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &DeformationGradient::identity()
                            )?,
                            &cauchy_tangent_stiffness_from_finite_difference_of_cauchy_stress(false)?
                        )
                    }
                }
            }
            mod first_piola_kirchhoff_stress
            {
                use super::*;
                mod deformed
                {
                    use super::*;
                    #[test]
                    fn finite_difference() -> Result<(), TestError>
                    {
                        assert_eq_from_fd(
                            &first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &get_deformation_gradient()
                            )?,
                            &first_piola_kirchhoff_tangent_stiffness_from_finite_difference_of_first_piola_kirchhoff_stress(true)?
                        )
                    }
                }
                mod undeformed
                {
                    use super::*;
                    #[test]
                    fn finite_difference() -> Result<(), TestError>
                    {
                        assert_eq_from_fd(
                            &first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &DeformationGradient::identity()
                            )?,
                            &first_piola_kirchhoff_tangent_stiffness_from_finite_difference_of_first_piola_kirchhoff_stress(false)?
                        )
                    }
                }
            }
            mod second_piola_kirchhoff_stress
            {
                use super::*;
                mod deformed
                {
                    use super::*;
                    #[test]
                    fn finite_difference() -> Result<(), TestError>
                    {
                        assert_eq_from_fd(
                            &second_piola_kirchhoff_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &get_deformation_gradient()
                            )?,
                            &second_piola_kirchhoff_tangent_stiffness_from_finite_difference_of_second_piola_kirchhoff_stress(true)?
                        )
                    }
                }
                mod undeformed
                {
                    use super::*;
                    #[test]
                    fn finite_difference() -> Result<(), TestError>
                    {
                        assert_eq_from_fd(
                            &second_piola_kirchhoff_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &DeformationGradient::identity()
                            )?,
                            &second_piola_kirchhoff_tangent_stiffness_from_finite_difference_of_second_piola_kirchhoff_stress(false)?
                        )
                    }
                }
            }
            mod cauchy_tangent_stiffness
            {
                use super::*;
                mod deformed
                {
                    use super::*;
                    #[test]
                    #[should_panic(expected = "Invalid Jacobian")]
                    fn invalid_jacobian()
                    {
                        let mut deformation_gradient = DeformationGradient::identity();
                        deformation_gradient[0][0] *= -1.0;
                        cauchy_tangent_stiffness_from_deformation_gradient!(
                            $constitutive_model, &deformation_gradient
                        ).unwrap();
                    }
                    #[test]
                    fn objectivity() -> Result<(), TestError>
                    {
                        assert_eq_within_tols(
                            &cauchy_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &get_deformation_gradient()
                            )?,
                            &cauchy_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &get_deformation_gradient_rotated()
                            )?.contract_all_indices_with_first_indices_of(
                                &get_rotation_current_configuration(),
                                &get_rotation_current_configuration(),
                                &get_rotation_current_configuration(),
                                &get_rotation_reference_configuration()
                            )
                        )
                    }
                    #[test]
                    fn symmetry() -> Result<(), TestError>
                    {
                        let cauchy_tangent_stiffness =
                        cauchy_tangent_stiffness_from_deformation_gradient!(
                            &$constitutive_model, &get_deformation_gradient()
                        )?;
                        assert_eq_within_tols(
                            &cauchy_tangent_stiffness,
                            &(0..3).map(|i|
                                (0..3).map(|j|
                                    cauchy_tangent_stiffness[j][i].clone()
                                ).collect()
                            ).collect()
                        )
                    }
                }
                mod undeformed
                {
                    use super::*;
                    #[test]
                    fn objectivity() -> Result<(), TestError>
                    {
                        assert_eq_within_tols(
                            &cauchy_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &DeformationGradient::identity()
                            )?,
                            &cauchy_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &get_deformation_gradient_rotated_undeformed()
                            )?.contract_all_indices_with_first_indices_of(
                                &get_rotation_current_configuration(),
                                &get_rotation_current_configuration(),
                                &get_rotation_current_configuration(),
                                &get_rotation_reference_configuration()
                            )
                        )
                    }
                    #[test]
                    fn symmetry() -> Result<(), TestError>
                    {
                        let cauchy_tangent_stiffness =
                        cauchy_tangent_stiffness_from_deformation_gradient!(
                            &$constitutive_model, &DeformationGradient::identity()
                        )?;
                        assert_eq_within_tols(
                            &cauchy_tangent_stiffness,
                            &(0..3).map(|i|
                                (0..3).map(|j|
                                    cauchy_tangent_stiffness[j][i].clone()
                                ).collect()
                            ).collect()
                        )
                    }
                }
            }
            mod first_piola_kirchhoff_tangent_stiffness
            {
                use super::*;
                mod deformed
                {
                    use super::*;
                    #[test]
                    #[should_panic(expected = "Invalid Jacobian")]
                    fn invalid_jacobian()
                    {
                        let mut deformation_gradient = DeformationGradient::identity();
                        deformation_gradient[0][0] *= -1.0;
                        first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient!(
                            $constitutive_model, &deformation_gradient
                        ).unwrap();
                    }
                    #[test]
                    fn objectivity() -> Result<(), TestError>
                    {
                        assert_eq_within_tols(
                            &first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &get_deformation_gradient()
                            )?,
                            &first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &get_deformation_gradient_rotated()
                            )?.contract_all_indices_with_first_indices_of(
                                &get_rotation_current_configuration(),
                                &get_rotation_reference_configuration(),
                                &get_rotation_current_configuration(),
                                &get_rotation_reference_configuration()
                            )
                        )
                    }
                }
                mod undeformed
                {
                    use super::*;
                    #[test]
                    fn objectivity() -> Result<(), TestError>
                    {
                        assert_eq_within_tols(
                            &first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &DeformationGradient::identity()
                            )?,
                            &first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &get_deformation_gradient_rotated_undeformed()
                            )?.contract_all_indices_with_first_indices_of(
                                &get_rotation_current_configuration(),
                                &get_rotation_reference_configuration(),
                                &get_rotation_current_configuration(),
                                &get_rotation_reference_configuration()
                            )
                        )
                    }
                }
            }
            mod second_piola_kirchhoff_tangent_stiffness
            {
                use super::*;
                mod deformed
                {
                    use super::*;
                    #[test]
                    #[should_panic(expected = "Invalid Jacobian")]
                    fn invalid_jacobian()
                    {
                        let mut deformation_gradient = DeformationGradient::identity();
                        deformation_gradient[0][0] *= -1.0;
                        second_piola_kirchhoff_tangent_stiffness_from_deformation_gradient!(
                            $constitutive_model, &deformation_gradient
                        ).unwrap();
                    }
                    #[test]
                    fn objectivity() -> Result<(), TestError>
                    {
                        assert_eq_within_tols(
                            &second_piola_kirchhoff_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &get_deformation_gradient()
                            )?,
                            &second_piola_kirchhoff_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &get_deformation_gradient_rotated()
                            )?.contract_all_indices_with_first_indices_of(
                                &get_rotation_reference_configuration(),
                                &get_rotation_reference_configuration(),
                                &get_rotation_current_configuration(),
                                &get_rotation_reference_configuration()
                            )
                        )
                    }
                    #[test]
                    fn symmetry() -> Result<(), TestError>
                    {
                        let second_piola_kirchhoff_tangent_stiffness =
                        second_piola_kirchhoff_tangent_stiffness_from_deformation_gradient!(
                            &$constitutive_model, &get_deformation_gradient()
                        )?;
                        assert_eq_within_tols(
                            &second_piola_kirchhoff_tangent_stiffness,
                            &(0..3).map(|i|
                                (0..3).map(|j|
                                    second_piola_kirchhoff_tangent_stiffness[j][i].clone()
                                ).collect()
                            ).collect()
                        )
                    }
                }
                mod undeformed
                {
                    use super::*;
                    #[test]
                    fn objectivity() -> Result<(), TestError>
                    {
                        assert_eq_within_tols(
                            &second_piola_kirchhoff_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &DeformationGradient::identity()
                            )?,
                            &second_piola_kirchhoff_tangent_stiffness_from_deformation_gradient!(
                                &$constitutive_model, &get_deformation_gradient_rotated_undeformed()
                            )?.contract_all_indices_with_first_indices_of(
                                &get_rotation_reference_configuration(),
                                &get_rotation_reference_configuration(),
                                &get_rotation_current_configuration(),
                                &get_rotation_reference_configuration()
                            )
                        )
                    }
                    #[test]
                    fn symmetry() -> Result<(), TestError>
                    {
                        let second_piola_kirchhoff_tangent_stiffness =
                        second_piola_kirchhoff_tangent_stiffness_from_deformation_gradient!(
                            &$constitutive_model, &DeformationGradient::identity()
                        )?;
                        assert_eq_within_tols(
                            &second_piola_kirchhoff_tangent_stiffness,
                            &(0..3).map(|i|
                                (0..3).map(|j|
                                    second_piola_kirchhoff_tangent_stiffness[j][i].clone()
                                ).collect()
                            ).collect()
                        )
                    }
                }
            }
        }
    }
}
pub(crate) use test_solid_constitutive_model_tangents;

macro_rules! test_solid_elastic_constitutive_model_no_root
{
    ($constitutive_model: expr) =>
    {
        crate::constitutive::solid::elastic::test::test_solid_constitutive_model!(
            $constitutive_model
        );
        mod elastic
        {
            use super::*;
            mod first_piola_kirchhoff_tangent_stiffness
            {
                use super::*;
                mod deformed
                {
                    use super::*;
                    #[test]
                    fn non_symmetry() -> Result<(), TestError>
                    {
                        let first_piola_kirchhoff_tangent_stiffness =
                        first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient!(
                            &$constitutive_model, &get_deformation_gradient()
                        )?;
                        assert!(
                            assert_eq_within_tols(
                                &first_piola_kirchhoff_tangent_stiffness,
                                &(0..3).map(|i|
                                    (0..3).map(|j|
                                        (0..3).map(|k|
                                            (0..3).map(|l|
                                                first_piola_kirchhoff_tangent_stiffness[k][l][i][j].clone()
                                            ).collect()
                                        ).collect()
                                    ).collect()
                                ).collect()
                            ).is_err()
                        );
                        Ok(())
                    }
                }
            }
        }
    }
}
pub(crate) use test_solid_elastic_constitutive_model_no_root;

macro_rules! test_solid_elastic_constitutive_model {
    ($constitutive_model: expr) => {
        crate::constitutive::solid::elastic::test::test_solid_elastic_constitutive_model_no_root!(
            $constitutive_model
        );
        crate::constitutive::solid::elastic::test::test_root!($constitutive_model);
    };
}
pub(crate) use test_solid_elastic_constitutive_model;

macro_rules! test_root {
    ($constitutive_model_constructed: expr) => {
        use crate::{constitutive::solid::elastic::AppliedLoad, math::Tensor};
        macro_rules! test_root_with_solver {
            ($solver: expr) => {
                #[test]
                fn uniaxial_compression() -> Result<(), crate::math::test::TestError> {
                    let deformation_gradient = $constitutive_model_constructed
                        .root(AppliedLoad::UniaxialStress(0.77), $solver)?;
                    let cauchy_stress =
                        $constitutive_model_constructed.cauchy_stress(&deformation_gradient)?;
                    assert!(cauchy_stress[0][0] < 0.0);
                    crate::math::test::assert_eq_within_tols(
                        &(cauchy_stress[1][1] / cauchy_stress[0][0]),
                        &0.0,
                    )?;
                    crate::math::test::assert_eq_within_tols(
                        &(cauchy_stress[2][2] / cauchy_stress[0][0]),
                        &0.0,
                    )?;
                    assert!(cauchy_stress.is_diagonal());
                    crate::math::test::assert_eq(
                        &deformation_gradient[1][1],
                        &deformation_gradient[2][2],
                    )?;
                    assert!(deformation_gradient.is_diagonal());
                    Ok(())
                }
                #[test]
                fn uniaxial_tension() -> Result<(), crate::math::test::TestError> {
                    let deformation_gradient = $constitutive_model_constructed
                        .root(AppliedLoad::UniaxialStress(1.2), $solver)?;
                    let cauchy_stress =
                        $constitutive_model_constructed.cauchy_stress(&deformation_gradient)?;
                    assert!(cauchy_stress[0][0] > 0.0);
                    assert!(cauchy_stress.is_diagonal());
                    crate::math::test::assert_eq_within_tols(&cauchy_stress[1][1], &0.0)?;
                    crate::math::test::assert_eq_within_tols(&cauchy_stress[2][2], &0.0)?;
                    assert!(deformation_gradient.is_diagonal());
                    crate::math::test::assert_eq(
                        &deformation_gradient[1][1],
                        &deformation_gradient[2][2],
                    )
                }
                #[test]
                fn uniaxial_undeformed() -> Result<(), crate::math::test::TestError> {
                    let deformation_gradient = $constitutive_model_constructed
                        .root(AppliedLoad::UniaxialStress(1.0), $solver)?;
                    let cauchy_stress =
                        $constitutive_model_constructed.cauchy_stress(&deformation_gradient)?;
                    assert!(cauchy_stress.is_zero());
                    assert!(deformation_gradient.is_identity());
                    Ok(())
                }
                #[test]
                fn biaxial_compression() -> Result<(), crate::math::test::TestError> {
                    let deformation_gradient = $constitutive_model_constructed
                        .root(AppliedLoad::BiaxialStress(0.77, 0.88), $solver)?;
                    let cauchy_stress =
                        $constitutive_model_constructed.cauchy_stress(&deformation_gradient)?;
                    assert!(cauchy_stress[0][0] < 0.0);
                    assert!(cauchy_stress[1][1] < 0.0);
                    crate::math::test::assert_eq_within_tols(
                        &(cauchy_stress[2][2]
                            / (cauchy_stress[0][0].powi(2) + cauchy_stress[1][1].powi(2)).sqrt()),
                        &0.0,
                    )?;
                    assert!(cauchy_stress.is_diagonal());
                    assert!(deformation_gradient.is_diagonal());
                    Ok(())
                }
                #[test]
                fn biaxial_mixed() -> Result<(), crate::math::test::TestError> {
                    let deformation_gradient = $constitutive_model_constructed
                        .root(AppliedLoad::BiaxialStress(1.3, 0.64), $solver)?;
                    let cauchy_stress =
                        $constitutive_model_constructed.cauchy_stress(&deformation_gradient)?;
                    assert!(cauchy_stress[0][0] > cauchy_stress[1][1]);
                    crate::math::test::assert_eq_within_tols(&cauchy_stress[2][2], &0.0)?;
                    assert!(cauchy_stress.is_diagonal());
                    assert!(deformation_gradient.is_diagonal());
                    Ok(())
                }
                #[test]
                fn biaxial_tension() -> Result<(), crate::math::test::TestError> {
                    let deformation_gradient = $constitutive_model_constructed
                        .root(AppliedLoad::BiaxialStress(1.3, 1.2), $solver)?;
                    let cauchy_stress =
                        $constitutive_model_constructed.cauchy_stress(&deformation_gradient)?;
                    assert!(cauchy_stress[0][0] > cauchy_stress[1][1]);
                    assert!(cauchy_stress[1][1] > 0.0);
                    crate::math::test::assert_eq_within_tols(&cauchy_stress[2][2], &0.0)?;
                    assert!(cauchy_stress.is_diagonal());
                    assert!(deformation_gradient.is_diagonal());
                    Ok(())
                }
                #[test]
                fn biaxial_undeformed() -> Result<(), crate::math::test::TestError> {
                    let deformation_gradient = $constitutive_model_constructed
                        .root(AppliedLoad::BiaxialStress(1.0, 1.0), $solver)?;
                    let cauchy_stress =
                        $constitutive_model_constructed.cauchy_stress(&deformation_gradient)?;
                    assert!(cauchy_stress.is_zero());
                    assert!(deformation_gradient.is_identity());
                    Ok(())
                }
            };
        }
        mod root {
            use super::*;
            mod gradient_descent {
                use super::*;
                use crate::{
                    constitutive::solid::elastic::ZerothOrderRoot, math::optimize::GradientDescent,
                };
                // test_root_with_solver!(GradientDescent::default());
                mod dual {
                    use super::*;
                    test_root_with_solver!(GradientDescent {
                        dual: true,
                        ..Default::default()
                    });
                }
            }
            mod newton_raphson {
                use super::*;
                use crate::{
                    constitutive::solid::elastic::FirstOrderRoot, math::optimize::NewtonRaphson,
                };
                test_root_with_solver!(NewtonRaphson::default());
            }
        }
    };
}
pub(crate) use test_root;
