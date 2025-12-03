pub use crate::{
    constitutive::solid::elastic::test::{BULK_MODULUS, SHEAR_MODULUS},
    math::Tensor,
    mechanics::Scalar,
};

pub const EXPONENT: Scalar = 1.1;
pub const EXTENSIBILITY: Scalar = 23.0;
pub const EXTRA_MODULUS: Scalar = 1.2;
pub const NUMBER_OF_LINKS: Scalar = 8.0;
pub const NUM_YEOH_EXTRA_MODULI: usize = 4;
pub const YEOH_EXTRA_MODULI: [Scalar; NUM_YEOH_EXTRA_MODULI] = [-1.0, 3e-1, -1e-3, 1e-5];

macro_rules! helmholtz_free_energy_density_from_deformation_gradient_simple {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.helmholtz_free_energy_density($deformation_gradient)
    };
}
pub(crate) use helmholtz_free_energy_density_from_deformation_gradient_simple;

macro_rules! use_elastic_macros {
    () => {
        use crate::constitutive::solid::elastic::test::{
            cauchy_stress_from_deformation_gradient,
            cauchy_stress_from_deformation_gradient_rotated,
            cauchy_stress_from_deformation_gradient_simple,
            cauchy_tangent_stiffness_from_deformation_gradient,
            first_piola_kirchhoff_stress_from_deformation_gradient,
            first_piola_kirchhoff_stress_from_deformation_gradient_rotated,
            first_piola_kirchhoff_stress_from_deformation_gradient_simple,
            first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient,
            first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient_simple,
            second_piola_kirchhoff_stress_from_deformation_gradient,
            second_piola_kirchhoff_stress_from_deformation_gradient_rotated,
            second_piola_kirchhoff_stress_from_deformation_gradient_simple,
            second_piola_kirchhoff_tangent_stiffness_from_deformation_gradient,
        };
    };
}
pub(crate) use use_elastic_macros;

macro_rules! use_elastic_macros_no_tangents {
    () => {
        use crate::constitutive::solid::elastic::test::{
            cauchy_stress_from_deformation_gradient,
            cauchy_stress_from_deformation_gradient_rotated,
            cauchy_stress_from_deformation_gradient_simple,
            first_piola_kirchhoff_stress_from_deformation_gradient,
            first_piola_kirchhoff_stress_from_deformation_gradient_rotated,
            first_piola_kirchhoff_stress_from_deformation_gradient_simple,
            second_piola_kirchhoff_stress_from_deformation_gradient,
            second_piola_kirchhoff_stress_from_deformation_gradient_rotated,
            second_piola_kirchhoff_stress_from_deformation_gradient_simple,
        };
    };
}
pub(crate) use use_elastic_macros_no_tangents;

macro_rules! test_solid_hyperelastic_constitutive_model_no_minimize
{
    ($constitutive_model: expr) =>
    {
        crate::constitutive::solid::elastic::test::test_solid_constitutive!(
            $constitutive_model
        );
        crate::constitutive::solid::hyperelastic::test::test_solid_hyperelastic_constitutive_model_no_tangents!(
            $constitutive_model
        );
        crate::constitutive::solid::hyperelastic::test::test_solid_hyperelastic_constitutive_model_tangents!(
            $constitutive_model
        );
    }
}
pub(crate) use test_solid_hyperelastic_constitutive_model_no_minimize;

macro_rules! test_solid_hyperelastic_constitutive_model
{
    ($constitutive_model: expr) =>
    {
        use_elastic_macros!();
        crate::constitutive::solid::hyperelastic::test::test_solid_hyperelastic_constitutive_model_no_minimize!(
            $constitutive_model
        );
        crate::constitutive::solid::hyperelastic::test::test_minimize_and_root!(
            $constitutive_model
        );
    }
}
pub(crate) use test_solid_hyperelastic_constitutive_model;

macro_rules! test_solid_hyperelastic_constitutive_model_no_tangents
{
    ($constitutive_model: expr) =>
    {
        crate::constitutive::solid::elastic::test::test_solid_constitutive_model_no_tangents!(
            $constitutive_model
        );
        fn first_piola_kirchhoff_stress_from_finite_difference_of_helmholtz_free_energy_density(is_deformed: bool) -> Result<FirstPiolaKirchhoffStress, TestError>
        {
            let mut first_piola_kirchhoff_stress = FirstPiolaKirchhoffStress::zero();
            for i in 0..3
            {
                for j in 0..3
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
                    deformation_gradient_plus[i][j] += 0.5 * EPSILON;
                    let helmholtz_free_energy_density_plus =
                    helmholtz_free_energy_density_from_deformation_gradient_simple!(
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
                    deformation_gradient_minus[i][j] -= 0.5 * EPSILON;
                    let helmholtz_free_energy_density_minus =
                    helmholtz_free_energy_density_from_deformation_gradient_simple!(
                        $constitutive_model, &deformation_gradient_minus
                    )?;
                    first_piola_kirchhoff_stress[i][j] = (
                        helmholtz_free_energy_density_plus - helmholtz_free_energy_density_minus
                    )/EPSILON;
                }
            }
            Ok(first_piola_kirchhoff_stress)
        }
        mod helmholtz_free_energy_density
        {
        use crate::math::test::assert_eq_from_fd;
            use super::*;
            mod deformed
            {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), TestError>
                {
                    assert_eq_from_fd(
                        &first_piola_kirchhoff_stress_from_deformation_gradient_simple!(
                            $constitutive_model, &get_deformation_gradient()
                        )?,
                        &first_piola_kirchhoff_stress_from_finite_difference_of_helmholtz_free_energy_density(true)?
                    )
                }
                #[test]
                #[should_panic(expected = "Invalid Jacobian")]
                fn invalid_jacobian()
                {
                    let mut deformation_gradient = DeformationGradient::identity();
                    deformation_gradient[0][0] *= -1.0;
                    helmholtz_free_energy_density_from_deformation_gradient_simple!(
                        $constitutive_model, &deformation_gradient
                    ).unwrap();
                }
                #[test]
                fn minimized() -> Result<(), TestError>
                {
                    let first_piola_kirchhoff_stress =
                    first_piola_kirchhoff_stress_from_deformation_gradient_simple!(
                        $constitutive_model, &get_deformation_gradient()
                    )?;
                    let minimum =
                    helmholtz_free_energy_density_from_deformation_gradient_simple!(
                        $constitutive_model, &get_deformation_gradient()
                    )? - first_piola_kirchhoff_stress.full_contraction(
                        &get_deformation_gradient()
                    );
                    let mut perturbed_deformation_gradient = get_deformation_gradient();
                    (0..3).try_for_each(|i|
                        (0..3).try_for_each(|j|{
                            perturbed_deformation_gradient = get_deformation_gradient();
                            perturbed_deformation_gradient[i][j] += 0.5 * EPSILON;
                            assert!(
                                helmholtz_free_energy_density_from_deformation_gradient_simple!(
                                    $constitutive_model, &perturbed_deformation_gradient
                                )? - first_piola_kirchhoff_stress.full_contraction(
                                    &perturbed_deformation_gradient
                                ) > minimum
                            );
                            perturbed_deformation_gradient[i][j] -= EPSILON;
                            assert!(
                                helmholtz_free_energy_density_from_deformation_gradient_simple!(
                                    $constitutive_model, &perturbed_deformation_gradient
                                )? - first_piola_kirchhoff_stress.full_contraction(
                                    &perturbed_deformation_gradient
                                ) > minimum
                            );
                            Ok(())
                        })
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError>
                {
                    assert_eq_within_tols(
                        &helmholtz_free_energy_density_from_deformation_gradient_simple!(
                            $constitutive_model, &get_deformation_gradient()
                        )?,
                        &helmholtz_free_energy_density_from_deformation_gradient_simple!(
                            $constitutive_model, &get_deformation_gradient_rotated()
                        )?
                    )
                }
                #[test]
                fn positive() -> Result<(), TestError>
                {
                    assert!(
                        helmholtz_free_energy_density_from_deformation_gradient_simple!(
                            $constitutive_model, &get_deformation_gradient()
                        )? > 0.0
                    );
                    Ok(())
                }
            }
            mod undeformed
            {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), TestError>
                {
                    assert_eq_from_fd(
                        &first_piola_kirchhoff_stress_from_finite_difference_of_helmholtz_free_energy_density(false)?,
                        &FirstPiolaKirchhoffStress::zero(),
                    )
                }
                #[test]
                fn minimized() -> Result<(), TestError>
                {
                    let minimum =
                    helmholtz_free_energy_density_from_deformation_gradient_simple!(
                        $constitutive_model, &DeformationGradient::identity()
                    )?;
                    let mut perturbed_deformation_gradient = DeformationGradient::identity();
                    (0..3).try_for_each(|i|
                        (0..3).try_for_each(|j|{
                            perturbed_deformation_gradient = DeformationGradient::identity();
                            perturbed_deformation_gradient[i][j] += 0.5 * EPSILON;
                            assert!(
                                helmholtz_free_energy_density_from_deformation_gradient_simple!(
                                    $constitutive_model, &perturbed_deformation_gradient
                                )? > minimum
                            );
                            perturbed_deformation_gradient[i][j] -= EPSILON;
                            assert!(
                                helmholtz_free_energy_density_from_deformation_gradient_simple!(
                                    $constitutive_model, &perturbed_deformation_gradient
                                )? > minimum
                            );
                            Ok(())
                        })
                    )
                }
                #[test]
                fn zero() -> Result<(), TestError>
                {
                    assert_eq(
                        &helmholtz_free_energy_density_from_deformation_gradient_simple!(
                            $constitutive_model, &DeformationGradient::identity()
                        )?, &0.0
                    )
                }
            }
        }
    }
}
pub(crate) use test_solid_hyperelastic_constitutive_model_no_tangents;

macro_rules! test_solid_hyperelastic_constitutive_model_tangents
{
    ($constitutive_model: expr) =>
    {
        crate::constitutive::solid::elastic::test::test_solid_constitutive_model_tangents!(
            $constitutive_model
        );
        mod hyperelastic
        {
            use super::*;
            mod first_piola_kirchhoff_tangent_stiffness
            {
                use super::*;
                mod deformed
                {
                    use super::*;
                    #[test]
                    fn symmetry() -> Result<(), TestError>
                    {
                        let first_piola_kirchhoff_tangent_stiffness =
                        first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient_simple!(
                            $constitutive_model, &get_deformation_gradient()
                        )?;
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
                        )
                    }
                }
                mod undeformed
                {
                    use super::*;
                    #[test]
                    fn symmetry() -> Result<(), TestError>
                    {
                        let first_piola_kirchhoff_tangent_stiffness =
                        first_piola_kirchhoff_tangent_stiffness_from_deformation_gradient_simple!(
                            $constitutive_model, &DeformationGradient::identity()
                        )?;
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
                        )
                    }
                }
            }
        }
    }
}
pub(crate) use test_solid_hyperelastic_constitutive_model_tangents;

macro_rules! test_minimize_and_root {
    ($constitutive_model: expr) => {
        crate::constitutive::solid::elastic::test::test_root!($constitutive_model);
        use crate::constitutive::solid::hyperelastic::Hyperelastic;
        macro_rules! test_minimize_with_solver {
            ($solver: expr) => {
                #[test]
                fn uniaxial_compression() -> Result<(), crate::math::test::TestError> {
                    let deformation_gradient =
                        $constitutive_model.minimize(AppliedLoad::UniaxialStress(0.77), $solver)?;
                    let cauchy_stress = $constitutive_model.cauchy_stress(&deformation_gradient)?;
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
                    let deformation_gradient =
                        $constitutive_model.minimize(AppliedLoad::UniaxialStress(1.2), $solver)?;
                    let cauchy_stress = $constitutive_model.cauchy_stress(&deformation_gradient)?;
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
                    let deformation_gradient =
                        $constitutive_model.minimize(AppliedLoad::UniaxialStress(1.0), $solver)?;
                    let cauchy_stress = $constitutive_model.cauchy_stress(&deformation_gradient)?;
                    assert!(cauchy_stress.is_zero());
                    assert!(deformation_gradient.is_identity());
                    Ok(())
                }
                #[test]
                fn biaxial_compression() -> Result<(), crate::math::test::TestError> {
                    let deformation_gradient = $constitutive_model
                        .minimize(AppliedLoad::BiaxialStress(0.77, 0.88), $solver)?;
                    let cauchy_stress = $constitutive_model.cauchy_stress(&deformation_gradient)?;
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
                    let deformation_gradient = $constitutive_model
                        .minimize(AppliedLoad::BiaxialStress(1.3, 0.64), $solver)?;
                    let cauchy_stress = $constitutive_model.cauchy_stress(&deformation_gradient)?;
                    assert!(cauchy_stress[0][0] > cauchy_stress[1][1]);
                    crate::math::test::assert_eq_within_tols(&cauchy_stress[2][2], &0.0)?;
                    assert!(cauchy_stress.is_diagonal());
                    assert!(deformation_gradient.is_diagonal());
                    Ok(())
                }
                #[test]
                fn biaxial_tension() -> Result<(), crate::math::test::TestError> {
                    let deformation_gradient = $constitutive_model
                        .minimize(AppliedLoad::BiaxialStress(1.3, 1.2), $solver)?;
                    let cauchy_stress = $constitutive_model.cauchy_stress(&deformation_gradient)?;
                    assert!(cauchy_stress[0][0] > cauchy_stress[1][1]);
                    assert!(cauchy_stress[1][1] > 0.0);
                    crate::math::test::assert_eq_within_tols(&cauchy_stress[2][2], &0.0)?;
                    assert!(cauchy_stress.is_diagonal());
                    assert!(deformation_gradient.is_diagonal());
                    Ok(())
                }
                #[test]
                fn biaxial_undeformed() -> Result<(), crate::math::test::TestError> {
                    let deformation_gradient = $constitutive_model
                        .minimize(AppliedLoad::BiaxialStress(1.0, 1.0), $solver)?;
                    let cauchy_stress = $constitutive_model.cauchy_stress(&deformation_gradient)?;
                    assert!(cauchy_stress.is_zero());
                    assert!(deformation_gradient.is_identity());
                    Ok(())
                }
            };
        }
        mod minimize {
            use super::*;
            mod gradient_descent {
                use super::*;
                use crate::{
                    constitutive::solid::hyperelastic::FirstOrderMinimize,
                    math::optimize::GradientDescent,
                };
                // test_minimize_with_solver!(GradientDescent::default());
                mod dual {
                    use super::*;
                    test_minimize_with_solver!(GradientDescent {
                        dual: true,
                        ..Default::default()
                    });
                }
            }
            mod newton_raphson {
                use super::*;
                use crate::{
                    constitutive::solid::hyperelastic::SecondOrderMinimize,
                    math::optimize::NewtonRaphson,
                };
                test_minimize_with_solver!(NewtonRaphson::default());
            }
        }
    };
}
pub(crate) use test_minimize_and_root;
