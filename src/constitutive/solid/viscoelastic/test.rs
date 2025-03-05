macro_rules! cauchy_stress_from_deformation_gradient {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr) => {
        $constitutive_model_constructed
            .cauchy_stress($deformation_gradient, &get_deformation_gradient_rate())
    };
}
pub(crate) use cauchy_stress_from_deformation_gradient;

macro_rules! cauchy_stress_from_deformation_gradient_simple {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr) => {
        $constitutive_model_constructed
            .cauchy_stress($deformation_gradient, &DeformationGradientRate::zero())
    };
}
pub(crate) use cauchy_stress_from_deformation_gradient_simple;

macro_rules! cauchy_stress_from_deformation_gradient_rotated {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr) => {
        $constitutive_model_constructed.cauchy_stress(
            $deformation_gradient,
            &get_deformation_gradient_rate_rotated(),
        )
    };
}
pub(crate) use cauchy_stress_from_deformation_gradient_rotated;

macro_rules! first_piola_kirchhoff_stress_from_deformation_gradient {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr) => {
        $constitutive_model_constructed
            .first_piola_kirchhoff_stress($deformation_gradient, &get_deformation_gradient_rate())
    };
}
pub(crate) use first_piola_kirchhoff_stress_from_deformation_gradient;

macro_rules! first_piola_kirchhoff_stress_from_deformation_gradient_simple {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr) => {
        $constitutive_model_constructed
            .first_piola_kirchhoff_stress($deformation_gradient, &DeformationGradientRate::zero())
    };
}
pub(crate) use first_piola_kirchhoff_stress_from_deformation_gradient_simple;

macro_rules! first_piola_kirchhoff_stress_from_deformation_gradient_rotated {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr) => {
        $constitutive_model_constructed.first_piola_kirchhoff_stress(
            $deformation_gradient,
            &get_deformation_gradient_rate_rotated(),
        )
    };
}
pub(crate) use first_piola_kirchhoff_stress_from_deformation_gradient_rotated;

macro_rules! second_piola_kirchhoff_stress_from_deformation_gradient {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr) => {
        $constitutive_model_constructed
            .second_piola_kirchhoff_stress($deformation_gradient, &get_deformation_gradient_rate())
    };
}
pub(crate) use second_piola_kirchhoff_stress_from_deformation_gradient;

macro_rules! second_piola_kirchhoff_stress_from_deformation_gradient_simple {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr) => {
        $constitutive_model_constructed
            .second_piola_kirchhoff_stress($deformation_gradient, &DeformationGradientRate::zero())
    };
}
pub(crate) use second_piola_kirchhoff_stress_from_deformation_gradient_simple;

macro_rules! second_piola_kirchhoff_stress_from_deformation_gradient_rotated {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr) => {
        $constitutive_model_constructed.second_piola_kirchhoff_stress(
            $deformation_gradient,
            &get_deformation_gradient_rate_rotated(),
        )
    };
}
pub(crate) use second_piola_kirchhoff_stress_from_deformation_gradient_rotated;

macro_rules! first_piola_kirchhoff_stress_from_deformation_gradient_rate_simple {
    ($constitutive_model_constructed: expr, $deformation_gradient_rate: expr) => {
        $constitutive_model_constructed.first_piola_kirchhoff_stress(
            &DeformationGradient::identity(),
            $deformation_gradient_rate,
        )
    };
}
pub(crate) use first_piola_kirchhoff_stress_from_deformation_gradient_rate_simple;

macro_rules! cauchy_stress_from_deformation_gradient_and_deformation_gradient_rate {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr, $deformation_gradient_rate: expr) => {
        $constitutive_model_constructed
            .cauchy_stress($deformation_gradient, $deformation_gradient_rate)
    };
}
pub(crate) use cauchy_stress_from_deformation_gradient_and_deformation_gradient_rate;

macro_rules! first_piola_kirchhoff_stress_from_deformation_gradient_and_deformation_gradient_rate {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr, $deformation_gradient_rate: expr) => {
        $constitutive_model_constructed
            .first_piola_kirchhoff_stress($deformation_gradient, $deformation_gradient_rate)
    };
}
pub(crate) use first_piola_kirchhoff_stress_from_deformation_gradient_and_deformation_gradient_rate;

macro_rules! second_piola_kirchhoff_stress_from_deformation_gradient_and_deformation_gradient_rate {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr, $deformation_gradient_rate: expr) => {
        $constitutive_model_constructed
            .second_piola_kirchhoff_stress($deformation_gradient, $deformation_gradient_rate)
    };
}
pub(crate) use second_piola_kirchhoff_stress_from_deformation_gradient_and_deformation_gradient_rate;

macro_rules! cauchy_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr, $deformation_gradient_rate: expr) => {
        $constitutive_model_constructed
            .cauchy_rate_tangent_stiffness($deformation_gradient, $deformation_gradient_rate)
    };
}
pub(crate) use cauchy_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate;

macro_rules! first_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr, $deformation_gradient_rate: expr) => {
        $constitutive_model_constructed.first_piola_kirchhoff_rate_tangent_stiffness(
            $deformation_gradient,
            $deformation_gradient_rate,
        )
    };
}
pub(crate) use first_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate;

macro_rules! second_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr, $deformation_gradient_rate: expr) => {
        $constitutive_model_constructed.second_piola_kirchhoff_rate_tangent_stiffness(
            $deformation_gradient,
            $deformation_gradient_rate,
        )
    };
}
pub(crate) use second_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate;

macro_rules! test_solid_viscous_constitutive_model
{
    ($constitutive_model: ident, $constitutive_model_parameters: expr, $constitutive_model_constructed: expr) =>
    {
        use crate::
        {
            math::{ContractAllIndicesWithFirstIndicesOf, test::assert_eq_from_fd},
            mechanics::test::
            {
                get_deformation_gradient_rotated_undeformed,
                get_deformation_gradient_rate,
                get_deformation_gradient_rate_rotated,
                get_deformation_gradient_rate_rotated_undeformed
            }
        };
        #[test]
        fn bulk_viscosity() -> Result<(), TestError>
        {
            let model = get_constitutive_model();
            assert_eq(model.bulk_viscosity(), &$constitutive_model_parameters[2])?;
            let mut deformation_gradient_rate = DeformationGradientRate::zero();
            deformation_gradient_rate += DeformationGradientRate::identity()*(EPSILON/3.0);
            let first_piola_kirchhoff_stress = first_piola_kirchhoff_stress_from_deformation_gradient_rate_simple!(&model, &deformation_gradient_rate)?;
            assert!((3.0*EPSILON*model.bulk_viscosity()/first_piola_kirchhoff_stress.trace() - 1.0).abs() < EPSILON);
            Ok(())
        }
        #[test]
        fn shear_viscosity() -> Result<(), TestError>
        {
            let model = get_constitutive_model();
            assert_eq(model.shear_viscosity(), &$constitutive_model_parameters[3])?;
            let mut deformation_gradient_rate = DeformationGradientRate::zero();
            deformation_gradient_rate[0][1] = EPSILON;
            let first_piola_kirchhoff_stress = first_piola_kirchhoff_stress_from_deformation_gradient_rate_simple!(&model, &deformation_gradient_rate)?;
            assert!((EPSILON*model.shear_viscosity()/first_piola_kirchhoff_stress[0][1] - 1.0).abs() < EPSILON);
            Ok(())
        }
        mod solid_viscous
        {
            use super::*;
            fn cauchy_rate_tangent_stiffness_from_finite_difference_of_cauchy_stress(is_deformed: bool) -> Result<CauchyRateTangentStiffness, TestError>
            {
                let deformation_gradient =
                    if is_deformed
                    {
                        get_deformation_gradient()
                    }
                    else
                    {
                        DeformationGradient::identity()
                    };
                let mut cauchy_rate_tangent_stiffness = CauchyTangentStiffness::zero();
                for k in 0..3
                {
                    for l in 0..3
                    {
                        let mut deformation_gradient_rate_plus =
                            if is_deformed
                            {
                                get_deformation_gradient_rate()
                            }
                            else
                            {
                                DeformationGradientRate::zero()
                            };
                        deformation_gradient_rate_plus[k][l] += 0.5*EPSILON;
                        let cauchy_stress_plus =
                        cauchy_stress_from_deformation_gradient_and_deformation_gradient_rate!(
                            &$constitutive_model_constructed, &deformation_gradient, &deformation_gradient_rate_plus
                        )?;
                        let mut deformation_gradient_rate_minus =
                            if is_deformed
                            {
                                get_deformation_gradient_rate()
                            }
                            else
                            {
                                DeformationGradientRate::zero()
                            };
                        deformation_gradient_rate_minus[k][l] -= 0.5*EPSILON;
                        let cauchy_stress_minus =
                        cauchy_stress_from_deformation_gradient_and_deformation_gradient_rate!(
                            &$constitutive_model_constructed, &deformation_gradient, &deformation_gradient_rate_minus
                        )?;
                        for i in 0..3
                        {
                            for j in 0..3
                            {
                                cauchy_rate_tangent_stiffness[i][j][k][l] = (
                                    cauchy_stress_plus[i][j] - cauchy_stress_minus[i][j]
                                )/EPSILON;
                            }
                        }
                    }
                }
                Ok(cauchy_rate_tangent_stiffness)
            }
            fn first_piola_kirchhoff_rate_tangent_stiffness_from_finite_difference_of_first_piola_kirchhoff_stress(is_deformed: bool) -> Result<FirstPiolaKirchhoffRateTangentStiffness, TestError>
            {
                let deformation_gradient =
                    if is_deformed
                    {
                        get_deformation_gradient()
                    }
                    else
                    {
                        DeformationGradient::identity()
                    };
                let mut first_piola_kirchhoff_rate_tangent_stiffness = FirstPiolaKirchhoffTangentStiffness::zero();
                for k in 0..3
                {
                    for l in 0..3
                    {
                        let mut deformation_gradient_rate_plus =
                            if is_deformed
                            {
                                get_deformation_gradient_rate()
                            }
                            else
                            {
                                DeformationGradientRate::zero()
                            };
                        deformation_gradient_rate_plus[k][l] += 0.5*EPSILON;
                        let first_piola_kirchhoff_stress_plus =
                        first_piola_kirchhoff_stress_from_deformation_gradient_and_deformation_gradient_rate!(
                            &$constitutive_model_constructed, &deformation_gradient, &deformation_gradient_rate_plus
                        )?;
                        let mut deformation_gradient_rate_minus =
                            if is_deformed
                            {
                                get_deformation_gradient_rate()
                            }
                            else
                            {
                                DeformationGradientRate::zero()
                            };
                        deformation_gradient_rate_minus[k][l] -= 0.5*EPSILON;
                        let first_piola_kirchhoff_stress_minus =
                        first_piola_kirchhoff_stress_from_deformation_gradient_and_deformation_gradient_rate!(
                            &$constitutive_model_constructed, &deformation_gradient, &deformation_gradient_rate_minus
                        )?;
                        for i in 0..3
                        {
                            for j in 0..3
                            {
                                first_piola_kirchhoff_rate_tangent_stiffness[i][j][k][l] = (
                                    first_piola_kirchhoff_stress_plus[i][j] - first_piola_kirchhoff_stress_minus[i][j]
                                )/EPSILON;
                            }
                        }
                    }
                }
                Ok(first_piola_kirchhoff_rate_tangent_stiffness)
            }
            fn second_piola_kirchhoff_rate_tangent_stiffness_from_finite_difference_of_second_piola_kirchhoff_stress(is_deformed: bool) -> Result<SecondPiolaKirchhoffRateTangentStiffness, TestError>
            {
                let deformation_gradient =
                    if is_deformed
                    {
                        get_deformation_gradient()
                    }
                    else
                    {
                        DeformationGradient::identity()
                    };
                let mut second_piola_kirchhoff_rate_tangent_stiffness = SecondPiolaKirchhoffTangentStiffness::zero();
                for k in 0..3
                {
                    for l in 0..3
                    {
                        let mut deformation_gradient_rate_plus =
                            if is_deformed
                            {
                                get_deformation_gradient_rate()
                            }
                            else
                            {
                                DeformationGradientRate::zero()
                            };
                        deformation_gradient_rate_plus[k][l] += 0.5*EPSILON;
                        let second_piola_kirchhoff_stress_plus =
                        second_piola_kirchhoff_stress_from_deformation_gradient_and_deformation_gradient_rate!(
                            &$constitutive_model_constructed, &deformation_gradient, &deformation_gradient_rate_plus
                        )?;
                        let mut deformation_gradient_rate_minus =
                            if is_deformed
                            {
                                get_deformation_gradient_rate()
                            }
                            else
                            {
                                DeformationGradientRate::zero()
                            };
                        deformation_gradient_rate_minus[k][l] -= 0.5*EPSILON;
                        let second_piola_kirchhoff_stress_minus =
                        second_piola_kirchhoff_stress_from_deformation_gradient_and_deformation_gradient_rate!(
                            &$constitutive_model_constructed, &deformation_gradient, &deformation_gradient_rate_minus
                        )?;
                        for i in 0..3
                        {
                            for j in 0..3
                            {
                                second_piola_kirchhoff_rate_tangent_stiffness[i][j][k][l] = (
                                    second_piola_kirchhoff_stress_plus[i][j] - second_piola_kirchhoff_stress_minus[i][j]
                                )/EPSILON;
                            }
                        }
                    }
                }
                Ok(second_piola_kirchhoff_rate_tangent_stiffness)
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
                            &cauchy_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                &$constitutive_model_constructed, &get_deformation_gradient(), &get_deformation_gradient_rate()
                            )?,
                            &cauchy_rate_tangent_stiffness_from_finite_difference_of_cauchy_stress(true)?
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
                            &cauchy_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                &$constitutive_model_constructed, &DeformationGradient::identity(), &DeformationGradientRate::zero()
                            )?,
                            &cauchy_rate_tangent_stiffness_from_finite_difference_of_cauchy_stress(false)?
                        )
                    }
                }
            }
            mod cauchy_rate_tangent_stiffness
            {
                use super::*;
                mod deformed
                {
                    use super::*;
                    #[test]
                    fn objectivity() -> Result<(), TestError>
                    {
                        assert_eq_within_tols(
                            &cauchy_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                &$constitutive_model_constructed, &get_deformation_gradient(), &get_deformation_gradient_rate()
                            )?, &(
                                cauchy_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                    &$constitutive_model_constructed, &get_deformation_gradient_rotated(), &get_deformation_gradient_rate_rotated()
                                )?.contract_all_indices_with_first_indices_of(
                                    &get_rotation_current_configuration(),
                                    &get_rotation_current_configuration(),
                                    &get_rotation_current_configuration(),
                                    &get_rotation_reference_configuration()
                                )
                            )
                        )

                    }
                    #[test]
                    fn symmetry() -> Result<(), TestError>
                    {
                        let cauchy_rate_tangent_stiffness =
                        cauchy_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                            &$constitutive_model_constructed, &get_deformation_gradient(), &get_deformation_gradient_rate()
                        )?;
                        assert_eq_within_tols(
                            &cauchy_rate_tangent_stiffness,
                            &(0..3).map(|i|
                                (0..3).map(|j|
                                    cauchy_rate_tangent_stiffness[j][i].clone()
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
                            &cauchy_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                &$constitutive_model_constructed, &DeformationGradient::identity(), &DeformationGradientRate::zero()
                            )?, &(
                                cauchy_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                    &$constitutive_model_constructed, &get_deformation_gradient_rotated_undeformed(), &get_deformation_gradient_rate_rotated_undeformed()
                                )?.contract_all_indices_with_first_indices_of(
                                    &get_rotation_current_configuration(),
                                    &get_rotation_current_configuration(),
                                    &get_rotation_current_configuration(),
                                    &get_rotation_reference_configuration()
                                )
                            )
                        )
                    }
                    #[test]
                    fn symmetry() -> Result<(), TestError>
                    {
                        let cauchy_rate_tangent_stiffness =
                        cauchy_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                            &$constitutive_model_constructed, &DeformationGradient::identity(), &DeformationGradientRate::zero()
                        )?;
                        assert_eq_within_tols(
                            &cauchy_rate_tangent_stiffness,
                            &(0..3).map(|i|
                                (0..3).map(|j|
                                    cauchy_rate_tangent_stiffness[j][i].clone()
                                ).collect()
                            ).collect()
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
                            &first_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                &$constitutive_model_constructed, &get_deformation_gradient(), &get_deformation_gradient_rate()
                            )?,
                            &first_piola_kirchhoff_rate_tangent_stiffness_from_finite_difference_of_first_piola_kirchhoff_stress(true)?
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
                            &first_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                &$constitutive_model_constructed, &DeformationGradient::identity(), &DeformationGradientRate::zero()
                            )?,
                            &first_piola_kirchhoff_rate_tangent_stiffness_from_finite_difference_of_first_piola_kirchhoff_stress(false)?
                        )
                    }
                }
            }
            mod first_piola_kirchhoff_rate_tangent_stiffness
            {
                use super::*;
                mod deformed
                {
                    use super::*;
                    #[test]
                    fn objectivity() -> Result<(), TestError>
                    {
                        assert_eq_within_tols(
                            &first_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                &$constitutive_model_constructed, &get_deformation_gradient(), &get_deformation_gradient_rate()
                            )?, &(
                                first_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                    &$constitutive_model_constructed, &get_deformation_gradient_rotated(), &get_deformation_gradient_rate_rotated()
                                )?.contract_all_indices_with_first_indices_of(
                                    &get_rotation_current_configuration(),
                                    &get_rotation_reference_configuration(),
                                    &get_rotation_current_configuration(),
                                    &get_rotation_reference_configuration()
                                )
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
                            &first_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                &$constitutive_model_constructed, &DeformationGradient::identity(), &DeformationGradientRate::zero()
                            )?, &(
                                first_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                    &$constitutive_model_constructed, &get_deformation_gradient_rotated_undeformed(), &get_deformation_gradient_rate_rotated_undeformed()
                                )?.contract_all_indices_with_first_indices_of(
                                    &get_rotation_current_configuration(),
                                    &get_rotation_reference_configuration(),
                                    &get_rotation_current_configuration(),
                                    &get_rotation_reference_configuration()
                                )
                            )
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
                            &second_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                &$constitutive_model_constructed, &get_deformation_gradient(), &get_deformation_gradient_rate()
                            )?,
                            &second_piola_kirchhoff_rate_tangent_stiffness_from_finite_difference_of_second_piola_kirchhoff_stress(true)?
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
                            &second_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                &$constitutive_model_constructed, &DeformationGradient::identity(), &DeformationGradientRate::zero()
                            )?,
                            &second_piola_kirchhoff_rate_tangent_stiffness_from_finite_difference_of_second_piola_kirchhoff_stress(false)?
                        )
                    }
                }
            }
            mod second_piola_kirchhoff_rate_tangent_stiffness
            {
                use super::*;
                mod deformed
                {
                    use super::*;
                    #[test]
                    fn objectivity() -> Result<(), TestError>
                    {
                        assert_eq_within_tols(
                            &second_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                &$constitutive_model_constructed, &get_deformation_gradient(), &get_deformation_gradient_rate()
                            )?, &(
                                second_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                    &$constitutive_model_constructed, &get_deformation_gradient_rotated(), &get_deformation_gradient_rate_rotated()
                                )?.contract_all_indices_with_first_indices_of(
                                    &get_rotation_reference_configuration(),
                                    &get_rotation_reference_configuration(),
                                    &get_rotation_current_configuration(),
                                    &get_rotation_reference_configuration()
                                )
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
                            &second_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                &$constitutive_model_constructed, &DeformationGradient::identity(), &DeformationGradientRate::zero()
                            )?, &(
                                second_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                                    &$constitutive_model_constructed, &get_deformation_gradient_rotated_undeformed(), &get_deformation_gradient_rate_rotated_undeformed()
                                )?.contract_all_indices_with_first_indices_of(
                                    &get_rotation_reference_configuration(),
                                    &get_rotation_reference_configuration(),
                                    &get_rotation_current_configuration(),
                                    &get_rotation_reference_configuration()
                                )
                            )
                        )
                    }
                }
            }
        }
    }
}
pub(crate) use test_solid_viscous_constitutive_model;

// macro_rules! test_solid_viscoelastic_constitutive_model
// {
//     ($constitutive_model: ident, $constitutive_model_parameters: expr, $constitutive_model_constructed: expr) =>
//     {
//         crate::constitutive::solid::elastic::test::test_solid_constitutive_model_no_tangents!(
//             $constitutive_model,
//             $constitutive_model_parameters,
//             $constitutive_model_constructed
//         );
//         crate::constitutive::solid::viscoelastic::test::test_solid_viscous_constitutive_model!(
//             $constitutive_model,
//             $constitutive_model_parameters,
//             $constitutive_model_constructed
//         );
//         mod viscoelastic
//         {
//             use crate::test::check_eq_within_tols;
//             use super::*;
//             mod first_piola_kirchhoff_rate_tangent_stiffness
//             {
//                 use super::*;
//                 mod deformed
//                 {
//                     use super::*;
//                     #[test]
//                     fn non_symmetry()
//                     {
//                         let first_piola_kirchhoff_rate_tangent_stiffness =
//                         first_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
//                             &$constitutive_model_constructed, &get_deformation_gradient(), &get_deformation_gradient_rate()
//                         );
//                         let mut sum: u8 = 0;
//                         for i in 0..3
//                         {
//                             for j in 0..3
//                             {
//                                 for k in 0..3
//                                 {
//                                     for l in 0..3
//                                     {
//                                         if check_eq_within_tols(
//                                             &first_piola_kirchhoff_rate_tangent_stiffness[i][j][k][l],
//                                             &first_piola_kirchhoff_rate_tangent_stiffness[k][l][i][j]
//                                         ) == false
//                                         {
//                                             sum += 1;
//                                         }
//                                     }
//                                 }
//                             }
//                         }
//                         assert!(sum > 0)
//                     }
//                 }
//             }
//         }
//     }
// }
// pub(crate) use test_solid_viscoelastic_constitutive_model;
