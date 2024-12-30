use crate::{constitutive::solid::elastic::test::ALMANSIHAMELPARAMETERS, mechanics::Scalar};

pub const ARRUDABOYCEPARAMETERS: &[Scalar; 3] =
    &[ALMANSIHAMELPARAMETERS[0], ALMANSIHAMELPARAMETERS[1], 8.0];
pub const FUNGPARAMETERS: &[Scalar; 4] = &[
    ALMANSIHAMELPARAMETERS[0],
    ALMANSIHAMELPARAMETERS[1],
    1.2,
    1.1,
];
pub const GENTPARAMETERS: &[Scalar; 3] =
    &[ALMANSIHAMELPARAMETERS[0], ALMANSIHAMELPARAMETERS[1], 23.0];
pub const MOONEYRIVLINPARAMETERS: &[Scalar; 3] =
    &[ALMANSIHAMELPARAMETERS[0], ALMANSIHAMELPARAMETERS[1], 1.1];
pub const NEOHOOKEANPARAMETERS: &[Scalar; 2] =
    &[ALMANSIHAMELPARAMETERS[0], ALMANSIHAMELPARAMETERS[1]];
pub const SAINTVENANTKIRCHOFFPARAMETERS: &[Scalar; 2] =
    &[ALMANSIHAMELPARAMETERS[0], ALMANSIHAMELPARAMETERS[1]];
pub const YEOHPARAMETERS: &[Scalar; 6] = &[
    ALMANSIHAMELPARAMETERS[0],
    ALMANSIHAMELPARAMETERS[1],
    -1.0,
    3e-1,
    -1e-3,
    1e-5,
];

macro_rules! test_solve {
    ($constitutive_model_constructed: expr) => {
        use crate::constitutive::solid::AppliedLoad;
        #[test]
        fn solve_biaxial_compression() -> Result<(), crate::math::test::TestError> {
            let (deformation_gradient, cauchy_stress) =
                $constitutive_model_constructed.solve(AppliedLoad::BiaxialStress(0.77, 0.88))?;
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
        fn solve_biaxial_mixed() -> Result<(), crate::math::test::TestError> {
            let (deformation_gradient, cauchy_stress) =
                $constitutive_model_constructed.solve(AppliedLoad::BiaxialStress(1.3, 0.44))?;
            assert!(cauchy_stress[0][0] > cauchy_stress[1][1]);
            crate::math::test::assert_eq_within_tols(&cauchy_stress[2][2], &0.0)?;
            assert!(cauchy_stress.is_diagonal());
            assert!(deformation_gradient.is_diagonal());
            Ok(())
        }
        #[test]
        fn solve_biaxial_tension() -> Result<(), crate::math::test::TestError> {
            let (deformation_gradient, cauchy_stress) =
                $constitutive_model_constructed.solve(AppliedLoad::BiaxialStress(1.3, 1.2))?;
            assert!(cauchy_stress[0][0] > cauchy_stress[1][1]);
            assert!(cauchy_stress[1][1] > 0.0);
            crate::math::test::assert_eq_within_tols(&cauchy_stress[2][2], &0.0)?;
            assert!(cauchy_stress.is_diagonal());
            assert!(deformation_gradient.is_diagonal());
            Ok(())
        }
        #[test]
        fn solve_biaxial_undeformed() -> Result<(), crate::math::test::TestError> {
            let (deformation_gradient, cauchy_stress) =
                $constitutive_model_constructed.solve(AppliedLoad::BiaxialStress(1.0, 1.0))?;
            assert!(cauchy_stress.is_zero());
            assert!(deformation_gradient.is_identity());
            Ok(())
        }
        #[test]
        fn solve_uniaxial_compression() -> Result<(), crate::math::test::TestError> {
            let (deformation_gradient, cauchy_stress) =
                $constitutive_model_constructed.solve(AppliedLoad::UniaxialStress(0.66))?;
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
            crate::math::test::assert_eq(&deformation_gradient[1][1], &deformation_gradient[2][2])?;
            assert!(deformation_gradient.is_diagonal());
            Ok(())
        }
        #[test]
        fn solve_uniaxial_tension() -> Result<(), crate::math::test::TestError> {
            let model = $constitutive_model_constructed;
            let (deformation_gradient, cauchy_stress) =
                model.solve(AppliedLoad::UniaxialStress(1.2))?;
            assert!(cauchy_stress[0][0] > 0.0);
            assert!(cauchy_stress.is_diagonal());
            crate::math::test::assert_eq_within_tols(&cauchy_stress[1][1], &0.0)?;
            crate::math::test::assert_eq_within_tols(&cauchy_stress[2][2], &0.0)?;
            assert!(deformation_gradient.is_diagonal());
            crate::math::test::assert_eq(&deformation_gradient[1][1], &deformation_gradient[2][2])
        }
        #[test]
        fn solve_uniaxial_undeformed() -> Result<(), crate::math::test::TestError> {
            let (deformation_gradient, cauchy_stress) =
                $constitutive_model_constructed.solve(AppliedLoad::UniaxialStress(1.0))?;
            assert!(cauchy_stress.is_zero());
            assert!(deformation_gradient.is_identity());
            Ok(())
        }
    };
}
pub(crate) use test_solve;

macro_rules! calculate_helmholtz_free_energy_density_from_deformation_gradient_simple {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr) => {
        $constitutive_model_constructed
            .calculate_helmholtz_free_energy_density($deformation_gradient)
    };
}
pub(crate) use calculate_helmholtz_free_energy_density_from_deformation_gradient_simple;

macro_rules! use_elastic_macros {
    () => {
        use crate::constitutive::solid::elastic::test::{
            calculate_cauchy_stress_from_deformation_gradient,
            calculate_cauchy_stress_from_deformation_gradient_rotated,
            calculate_cauchy_stress_from_deformation_gradient_simple,
            calculate_cauchy_tangent_stiffness_from_deformation_gradient,
            calculate_first_piola_kirchoff_stress_from_deformation_gradient,
            calculate_first_piola_kirchoff_stress_from_deformation_gradient_rotated,
            calculate_first_piola_kirchoff_stress_from_deformation_gradient_simple,
            calculate_first_piola_kirchoff_tangent_stiffness_from_deformation_gradient,
            calculate_first_piola_kirchoff_tangent_stiffness_from_deformation_gradient_simple,
            calculate_second_piola_kirchoff_stress_from_deformation_gradient,
            calculate_second_piola_kirchoff_stress_from_deformation_gradient_rotated,
            calculate_second_piola_kirchoff_stress_from_deformation_gradient_simple,
            calculate_second_piola_kirchoff_tangent_stiffness_from_deformation_gradient,
        };
    };
}
pub(crate) use use_elastic_macros;

macro_rules! use_elastic_macros_no_tangents {
    () => {
        use crate::constitutive::solid::elastic::test::{
            calculate_cauchy_stress_from_deformation_gradient,
            calculate_cauchy_stress_from_deformation_gradient_rotated,
            calculate_cauchy_stress_from_deformation_gradient_simple,
            calculate_first_piola_kirchoff_stress_from_deformation_gradient,
            calculate_first_piola_kirchoff_stress_from_deformation_gradient_rotated,
            calculate_first_piola_kirchoff_stress_from_deformation_gradient_simple,
            calculate_second_piola_kirchoff_stress_from_deformation_gradient,
            calculate_second_piola_kirchoff_stress_from_deformation_gradient_rotated,
            calculate_second_piola_kirchoff_stress_from_deformation_gradient_simple,
        };
    };
}
pub(crate) use use_elastic_macros_no_tangents;

macro_rules! test_solid_hyperelastic_constitutive_model
{
    ($constitutive_model: ident, $constitutive_model_parameters: expr, $constitutive_model_constructed: expr) =>
    {
        crate::constitutive::solid::elastic::test::test_solid_constitutive_construction!(
            $constitutive_model, $constitutive_model_parameters, $constitutive_model_constructed
        );
        crate::constitutive::solid::hyperelastic::test::test_constructed_solid_hyperelastic_constitutive_model!(
            $constitutive_model_constructed
        );
    }
}
pub(crate) use test_solid_hyperelastic_constitutive_model;

macro_rules! test_constructed_solid_hyperelastic_constitutive_model
{
    ($constitutive_model_constructed: expr) =>
    {
        crate::constitutive::solid::hyperelastic::test::test_solid_hyperelastic_constitutive_model_no_tangents!(
            $constitutive_model_constructed
        );
        crate::constitutive::solid::hyperelastic::test::test_solid_hyperelastic_constitutive_model_tangents!(
            $constitutive_model_constructed
        );
    }
}
pub(crate) use test_constructed_solid_hyperelastic_constitutive_model;

macro_rules! test_solid_hyperelastic_constitutive_model_no_tangents
{
    ($constitutive_model_constructed: expr) =>
    {
        crate::constitutive::solid::elastic::test::test_solid_constitutive_model_no_tangents!(
            $constitutive_model_constructed
        );
        fn calculate_first_piola_kirchoff_stress_from_finite_difference_of_helmholtz_free_energy_density(is_deformed: bool) -> Result<FirstPiolaKirchoffStress, TestError>
        {
            let mut first_piola_kirchoff_stress = FirstPiolaKirchoffStress::zero();
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
                    deformation_gradient_plus[i][j] += 0.5*EPSILON;
                    let helmholtz_free_energy_density_plus =
                    calculate_helmholtz_free_energy_density_from_deformation_gradient_simple!(
                        $constitutive_model_constructed, &deformation_gradient_plus
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
                    deformation_gradient_minus[i][j] -= 0.5*EPSILON;
                    let helmholtz_free_energy_density_minus =
                    calculate_helmholtz_free_energy_density_from_deformation_gradient_simple!(
                        $constitutive_model_constructed, &deformation_gradient_minus
                    )?;
                    first_piola_kirchoff_stress[i][j] = (
                        helmholtz_free_energy_density_plus - helmholtz_free_energy_density_minus
                    )/EPSILON;
                }
            }
            Ok(first_piola_kirchoff_stress)
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
                        &calculate_first_piola_kirchoff_stress_from_deformation_gradient_simple!(
                            $constitutive_model_constructed, &get_deformation_gradient()
                        )?,
                        &calculate_first_piola_kirchoff_stress_from_finite_difference_of_helmholtz_free_energy_density(true)?
                    )
                }
                #[test]
                #[should_panic(expected = "Invalid Jacobian")]
                fn invalid_jacobian()
                {
                    let mut deformation_gradient = DeformationGradient::identity();
                    deformation_gradient[0][0] *= -1.0;
                    calculate_helmholtz_free_energy_density_from_deformation_gradient_simple!(
                        $constitutive_model_constructed, &deformation_gradient
                    ).unwrap();
                }
                #[test]
                fn minimized() -> Result<(), TestError>
                {
                    let first_piola_kirchoff_stress =
                    calculate_first_piola_kirchoff_stress_from_deformation_gradient_simple!(
                        $constitutive_model_constructed, &get_deformation_gradient()
                    )?;
                    let minimum =
                    calculate_helmholtz_free_energy_density_from_deformation_gradient_simple!(
                        $constitutive_model_constructed, &get_deformation_gradient()
                    )? - first_piola_kirchoff_stress.full_contraction(
                        &get_deformation_gradient()
                    );
                    let mut perturbed_deformation_gradient = get_deformation_gradient();
                    (0..3).try_for_each(|i|
                        (0..3).try_for_each(|j|{
                            perturbed_deformation_gradient = get_deformation_gradient();
                            perturbed_deformation_gradient[i][j] += 0.5 * EPSILON;
                            assert!(
                                calculate_helmholtz_free_energy_density_from_deformation_gradient_simple!(
                                    $constitutive_model_constructed, &perturbed_deformation_gradient
                                )? - first_piola_kirchoff_stress.full_contraction(
                                    &perturbed_deformation_gradient
                                ) > minimum
                            );
                            perturbed_deformation_gradient[i][j] -= EPSILON;
                            assert!(
                                calculate_helmholtz_free_energy_density_from_deformation_gradient_simple!(
                                    $constitutive_model_constructed, &perturbed_deformation_gradient
                                )? - first_piola_kirchoff_stress.full_contraction(
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
                        &calculate_helmholtz_free_energy_density_from_deformation_gradient_simple!(
                            $constitutive_model_constructed,  &get_deformation_gradient()
                        )?,
                        &calculate_helmholtz_free_energy_density_from_deformation_gradient_simple!(
                            $constitutive_model_constructed,  &get_deformation_gradient_rotated()
                        )?
                    )
                }
                #[test]
                fn positive() -> Result<(), TestError>
                {
                    assert!(
                        calculate_helmholtz_free_energy_density_from_deformation_gradient_simple!(
                            $constitutive_model_constructed,  &get_deformation_gradient()
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
                        &calculate_first_piola_kirchoff_stress_from_finite_difference_of_helmholtz_free_energy_density(false)?,
                        &FirstPiolaKirchoffStress::zero(),
                    )
                }
                #[test]
                fn minimized() -> Result<(), TestError>
                {
                    let minimum =
                    calculate_helmholtz_free_energy_density_from_deformation_gradient_simple!(
                        $constitutive_model_constructed, &DeformationGradient::identity()
                    )?;
                    let mut perturbed_deformation_gradient = DeformationGradient::identity();
                    (0..3).try_for_each(|i|
                        (0..3).try_for_each(|j|{
                            perturbed_deformation_gradient = DeformationGradient::identity();
                            perturbed_deformation_gradient[i][j] += 0.5 * EPSILON;
                            assert!(
                                calculate_helmholtz_free_energy_density_from_deformation_gradient_simple!(
                                    $constitutive_model_constructed, &perturbed_deformation_gradient
                                )? > minimum
                            );
                            perturbed_deformation_gradient[i][j] -= EPSILON;
                            assert!(
                                calculate_helmholtz_free_energy_density_from_deformation_gradient_simple!(
                                    $constitutive_model_constructed, &perturbed_deformation_gradient
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
                        &calculate_helmholtz_free_energy_density_from_deformation_gradient_simple!(
                            $constitutive_model_constructed,  &DeformationGradient::identity()
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
    ($constitutive_model_constructed: expr) =>
    {
        crate::constitutive::solid::elastic::test::test_solid_constitutive_model_tangents!(
            $constitutive_model_constructed
        );
        mod hyperelastic
        {
            use super::*;
            mod first_piola_kirchoff_tangent_stiffness
            {
                use super::*;
                mod deformed
                {
                    use super::*;
                    #[test]
                    fn symmetry() -> Result<(), TestError>
                    {
                        let first_piola_kirchoff_tangent_stiffness =
                        calculate_first_piola_kirchoff_tangent_stiffness_from_deformation_gradient_simple!(
                            $constitutive_model_constructed, &get_deformation_gradient()
                        )?;
                        assert_eq_within_tols(
                            &first_piola_kirchoff_tangent_stiffness,
                            &(0..3).map(|i|
                                (0..3).map(|j|
                                    (0..3).map(|k|
                                        (0..3).map(|l|
                                            first_piola_kirchoff_tangent_stiffness[k][l][i][j].copy()
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
                        let first_piola_kirchoff_tangent_stiffness =
                        calculate_first_piola_kirchoff_tangent_stiffness_from_deformation_gradient_simple!(
                            $constitutive_model_constructed, &DeformationGradient::identity()
                        )?;
                        assert_eq_within_tols(
                            &first_piola_kirchoff_tangent_stiffness,
                            &(0..3).map(|i|
                                (0..3).map(|j|
                                    (0..3).map(|k|
                                        (0..3).map(|l|
                                            first_piola_kirchoff_tangent_stiffness[k][l][i][j].copy()
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
