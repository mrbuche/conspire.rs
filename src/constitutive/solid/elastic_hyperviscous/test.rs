use crate::{
    constitutive::solid::elastic::test::ALMANSIHAMELPARAMETERS as ALMANSIHAMELPARAMETERSELASTIC,
    mechanics::Scalar,
};
pub const ALMANSIHAMELPARAMETERS: &[Scalar; 4] = &[
    ALMANSIHAMELPARAMETERSELASTIC[0],
    ALMANSIHAMELPARAMETERSELASTIC[1],
    11.0,
    1.0,
];

macro_rules! test_minimize_and_root {
    ($constitutive_model_constructed: expr) => {
        use crate::constitutive::solid::viscoelastic::AppliedLoad;
        macro_rules! test_with_integrator_and_solver {
            ($integrator: ident, $solver: expr) => {
                #[test]
                fn minimize_uniaxial_compression() -> Result<(), TestError> {
                    let (_, deformation_gradients, deformation_gradient_rates) =
                        $constitutive_model_constructed.minimize(
                            AppliedLoad::UniaxialStress(|t: Scalar| -t, &[0.0, 1.0]),
                            $integrator::default(),
                            $solver,
                        )?;
                    deformation_gradients
                        .iter()
                        .skip(1)
                        .try_for_each(|deformation_gradient| {
                            assert!(deformation_gradient.is_diagonal());
                            assert!(deformation_gradient[0][0] < deformation_gradient[1][1]);
                            assert_eq_within_tols(
                                &deformation_gradient[1][1],
                                &deformation_gradient[2][2],
                            )
                        })?;
                    deformation_gradient_rates.iter().skip(1).try_for_each(
                        |deformation_gradient_rate| {
                            assert!(deformation_gradient_rate.is_diagonal());
                            assert!(
                                deformation_gradient_rate[0][0] < deformation_gradient_rate[1][1]
                            );
                            assert_eq_within_tols(
                                &deformation_gradient_rate[1][1],
                                &deformation_gradient_rate[2][2],
                            )
                        },
                    )
                }
                #[test]
                fn minimize_uniaxial_tension() -> Result<(), TestError> {
                    let (_, deformation_gradients, deformation_gradient_rates) =
                        $constitutive_model_constructed.minimize(
                            AppliedLoad::UniaxialStress(|t: Scalar| t, &[0.0, 1.0]),
                            $integrator::default(),
                            $solver,
                        )?;
                    deformation_gradients
                        .iter()
                        .skip(1)
                        .try_for_each(|deformation_gradient| {
                            assert!(deformation_gradient.is_diagonal());
                            assert!(deformation_gradient[0][0] > deformation_gradient[1][1]);
                            assert_eq_within_tols(
                                &deformation_gradient[1][1],
                                &deformation_gradient[2][2],
                            )
                        })?;
                    deformation_gradient_rates.iter().skip(1).try_for_each(
                        |deformation_gradient_rate| {
                            assert!(deformation_gradient_rate.is_diagonal());
                            assert!(
                                deformation_gradient_rate[0][0] > deformation_gradient_rate[1][1]
                            );
                            assert_eq_within_tols(
                                &deformation_gradient_rate[1][1],
                                &deformation_gradient_rate[2][2],
                            )
                        },
                    )
                }
                #[test]
                fn minimize_uniaxial_undeformed() -> Result<(), TestError> {
                    let (_, deformation_gradients, deformation_gradient_rates) =
                        $constitutive_model_constructed.minimize(
                            AppliedLoad::UniaxialStress(|_| 0.0, &[0.0, 1.0]),
                            $integrator::default(),
                            $solver,
                        )?;
                    deformation_gradients
                        .iter()
                        .try_for_each(|deformation_gradient| {
                            assert_eq_within_tols(
                                deformation_gradient,
                                &DeformationGradient::identity(),
                            )
                        })?;
                    deformation_gradient_rates
                        .iter()
                        .try_for_each(|deformation_gradient_rate| {
                            assert_eq_within_tols(
                                deformation_gradient_rate,
                                &DeformationGradientRate::zero(),
                            )
                        })
                }
                #[test]
                fn minimize_biaxial_tension() -> Result<(), TestError> {
                    let (_, deformation_gradients, deformation_gradient_rates) =
                        $constitutive_model_constructed.minimize(
                            AppliedLoad::BiaxialStress(
                                |t: Scalar| 0.3 * t,
                                |t: Scalar| 0.2 * t,
                                &[0.0, 1.0],
                            ),
                            $integrator::default(),
                            $solver,
                        )?;
                    deformation_gradients
                        .iter()
                        .skip(1)
                        .for_each(|deformation_gradient| {
                            assert!(deformation_gradient.is_diagonal());
                            assert!(deformation_gradient[0][0] > deformation_gradient[2][2]);
                            assert!(deformation_gradient[1][1] > deformation_gradient[2][2]);
                        });
                    deformation_gradient_rates.iter().skip(1).for_each(
                        |deformation_gradient_rate| {
                            assert!(deformation_gradient_rate.is_diagonal());
                            assert!(
                                deformation_gradient_rate[0][0] > deformation_gradient_rate[2][2]
                            );
                            assert!(
                                deformation_gradient_rate[1][1] > deformation_gradient_rate[2][2]
                            );
                        },
                    );
                    Ok(())
                }
                #[test]
                fn minimize_biaxial_undeformed() -> Result<(), TestError> {
                    let (_, deformation_gradients, deformation_gradient_rates) =
                        $constitutive_model_constructed.minimize(
                            AppliedLoad::BiaxialStress(|_| 0.0, |_| 0.0, &[0.0, 1.0]),
                            $integrator::default(),
                            $solver,
                        )?;
                    deformation_gradients
                        .iter()
                        .try_for_each(|deformation_gradient| {
                            assert_eq_within_tols(
                                deformation_gradient,
                                &DeformationGradient::identity(),
                            )
                        })?;
                    deformation_gradient_rates
                        .iter()
                        .try_for_each(|deformation_gradient_rate| {
                            assert_eq_within_tols(
                                deformation_gradient_rate,
                                &DeformationGradientRate::zero(),
                            )
                        })
                }
                #[test]
                fn root_uniaxial_compression() -> Result<(), TestError> {
                    let (_, deformation_gradients, deformation_gradient_rates) =
                        $constitutive_model_constructed.root(
                            AppliedLoad::UniaxialStress(|t: Scalar| -t, &[0.0, 1.0]),
                            $integrator::default(),
                            $solver,
                        )?;
                    deformation_gradients
                        .iter()
                        .skip(1)
                        .try_for_each(|deformation_gradient| {
                            assert!(deformation_gradient.is_diagonal());
                            assert!(deformation_gradient[0][0] < deformation_gradient[1][1]);
                            assert_eq_within_tols(
                                &deformation_gradient[1][1],
                                &deformation_gradient[2][2],
                            )
                        })?;
                    deformation_gradient_rates.iter().skip(1).try_for_each(
                        |deformation_gradient_rate| {
                            assert!(deformation_gradient_rate.is_diagonal());
                            assert!(
                                deformation_gradient_rate[0][0] < deformation_gradient_rate[1][1]
                            );
                            assert_eq_within_tols(
                                &deformation_gradient_rate[1][1],
                                &deformation_gradient_rate[2][2],
                            )
                        },
                    )
                }
                #[test]
                fn root_uniaxial_tension() -> Result<(), TestError> {
                    let (_, deformation_gradients, deformation_gradient_rates) =
                        $constitutive_model_constructed.root(
                            AppliedLoad::UniaxialStress(|t: Scalar| t, &[0.0, 1.0]),
                            $integrator::default(),
                            $solver,
                        )?;
                    deformation_gradients
                        .iter()
                        .skip(1)
                        .try_for_each(|deformation_gradient| {
                            assert!(deformation_gradient.is_diagonal());
                            assert!(deformation_gradient[0][0] > deformation_gradient[1][1]);
                            assert_eq_within_tols(
                                &deformation_gradient[1][1],
                                &deformation_gradient[2][2],
                            )
                        })?;
                    deformation_gradient_rates.iter().skip(1).try_for_each(
                        |deformation_gradient_rate| {
                            assert!(deformation_gradient_rate.is_diagonal());
                            assert!(
                                deformation_gradient_rate[0][0] > deformation_gradient_rate[1][1]
                            );
                            assert_eq_within_tols(
                                &deformation_gradient_rate[1][1],
                                &deformation_gradient_rate[2][2],
                            )
                        },
                    )
                }
                #[test]
                fn root_uniaxial_undeformed() -> Result<(), TestError> {
                    let (_, deformation_gradients, deformation_gradient_rates) =
                        $constitutive_model_constructed.root(
                            AppliedLoad::UniaxialStress(|_| 0.0, &[0.0, 1.0]),
                            $integrator::default(),
                            $solver,
                        )?;
                    deformation_gradients
                        .iter()
                        .try_for_each(|deformation_gradient| {
                            assert_eq_within_tols(
                                deformation_gradient,
                                &DeformationGradient::identity(),
                            )
                        })?;
                    deformation_gradient_rates
                        .iter()
                        .try_for_each(|deformation_gradient_rate| {
                            assert_eq_within_tols(
                                deformation_gradient_rate,
                                &DeformationGradientRate::zero(),
                            )
                        })
                }
                #[test]
                fn root_biaxial_tension() -> Result<(), TestError> {
                    let (_, deformation_gradients, deformation_gradient_rates) =
                        $constitutive_model_constructed.root(
                            AppliedLoad::BiaxialStress(
                                |t: Scalar| 0.3 * t,
                                |t: Scalar| 0.2 * t,
                                &[0.0, 1.0],
                            ),
                            $integrator::default(),
                            $solver,
                        )?;
                    deformation_gradients
                        .iter()
                        .skip(1)
                        .for_each(|deformation_gradient| {
                            assert!(deformation_gradient.is_diagonal());
                            assert!(deformation_gradient[0][0] > deformation_gradient[2][2]);
                            assert!(deformation_gradient[1][1] > deformation_gradient[2][2]);
                        });
                    deformation_gradient_rates.iter().skip(1).for_each(
                        |deformation_gradient_rate| {
                            assert!(deformation_gradient_rate.is_diagonal());
                            assert!(
                                deformation_gradient_rate[0][0] > deformation_gradient_rate[2][2]
                            );
                            assert!(
                                deformation_gradient_rate[1][1] > deformation_gradient_rate[2][2]
                            );
                        },
                    );
                    Ok(())
                }
                #[test]
                fn root_biaxial_undeformed() -> Result<(), TestError> {
                    let (_, deformation_gradients, deformation_gradient_rates) =
                        $constitutive_model_constructed.root(
                            AppliedLoad::BiaxialStress(|_| 0.0, |_| 0.0, &[0.0, 1.0]),
                            $integrator::default(),
                            $solver,
                        )?;
                    deformation_gradients
                        .iter()
                        .try_for_each(|deformation_gradient| {
                            assert_eq_within_tols(
                                deformation_gradient,
                                &DeformationGradient::identity(),
                            )
                        })?;
                    deformation_gradient_rates
                        .iter()
                        .try_for_each(|deformation_gradient_rate| {
                            assert_eq_within_tols(
                                deformation_gradient_rate,
                                &DeformationGradientRate::zero(),
                            )
                        })
                }
            };
        }
        macro_rules! test_with_solver {
            ($solver: expr) => {
                mod bogacki_shampine {
                    use super::*;
                    use crate::math::integrate::BogackiShampine;
                    test_with_integrator_and_solver!(BogackiShampine, $solver);
                }
                mod dormand_prince {
                    use super::*;
                    use crate::math::integrate::DormandPrince;
                    test_with_integrator_and_solver!(DormandPrince, $solver);
                }
                mod verner_8 {
                    use super::*;
                    use crate::math::integrate::Verner8;
                    test_with_integrator_and_solver!(Verner8, $solver);
                }
                mod verner_9 {
                    use super::*;
                    use crate::math::integrate::Verner9;
                    test_with_integrator_and_solver!(Verner9, $solver);
                }
                #[test]
                fn minimize_uniaxial_compression_inner() -> Result<(), TestError> {
                    let deformation_gradient_rate = $constitutive_model_constructed
                        .minimize_uniaxial_inner(
                            &DeformationGradient::identity(),
                            -4.4,
                            &$solver,
                            &DeformationGradientRate::zero(),
                        )?;
                    assert!(deformation_gradient_rate.is_diagonal());
                    assert!(deformation_gradient_rate[0][0] < deformation_gradient_rate[1][1]);
                    assert_eq_within_tols(
                        &deformation_gradient_rate[1][1],
                        &deformation_gradient_rate[2][2],
                    )
                }
                #[test]
                fn minimize_uniaxial_tension_inner() -> Result<(), TestError> {
                    let deformation_gradient_rate = $constitutive_model_constructed
                        .minimize_uniaxial_inner(
                            &DeformationGradient::identity(),
                            4.4,
                            &$solver,
                            &DeformationGradientRate::zero(),
                        )?;
                    assert!(deformation_gradient_rate.is_diagonal());
                    assert!(deformation_gradient_rate[0][0] > deformation_gradient_rate[1][1]);
                    assert_eq_within_tols(
                        &deformation_gradient_rate[1][1],
                        &deformation_gradient_rate[2][2],
                    )
                }
                #[test]
                fn minimize_uniaxial_undeformed_inner() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &$constitutive_model_constructed.minimize_uniaxial_inner(
                            &DeformationGradient::identity(),
                            0.0,
                            &$solver,
                            &DeformationGradientRate::zero(),
                        )?,
                        &DeformationGradientRate::zero(),
                    )
                }
                #[test]
                fn minimize_biaxial_compression_inner() -> Result<(), TestError> {
                    let deformation_gradient_rate = $constitutive_model_constructed
                        .minimize_biaxial_inner(
                            &DeformationGradient::identity(),
                            -0.77,
                            -0.88,
                            &$solver,
                            &DeformationGradientRate::zero(),
                        )?;
                    assert!(deformation_gradient_rate.is_diagonal());
                    assert!(deformation_gradient_rate[0][0] < deformation_gradient_rate[2][2]);
                    assert!(deformation_gradient_rate[1][1] < deformation_gradient_rate[2][2]);
                    Ok(())
                }
                #[test]
                fn minimize_biaxial_mixed_inner() -> Result<(), TestError> {
                    let deformation_gradient_rate = $constitutive_model_constructed
                        .minimize_biaxial_inner(
                            &DeformationGradient::identity(),
                            1.3,
                            -0.64,
                            &$solver,
                            &DeformationGradientRate::zero(),
                        )?;
                    assert!(deformation_gradient_rate.is_diagonal());
                    assert!(deformation_gradient_rate[0][0] > deformation_gradient_rate[2][2]);
                    assert!(deformation_gradient_rate[1][1] < deformation_gradient_rate[2][2]);
                    Ok(())
                }
                #[test]
                fn minimize_biaxial_tension_inner() -> Result<(), TestError> {
                    let deformation_gradient_rate = $constitutive_model_constructed
                        .minimize_biaxial_inner(
                            &DeformationGradient::identity(),
                            1.3,
                            1.2,
                            &$solver,
                            &DeformationGradientRate::zero(),
                        )?;
                    assert!(deformation_gradient_rate.is_diagonal());
                    assert!(deformation_gradient_rate[0][0] > deformation_gradient_rate[2][2]);
                    assert!(deformation_gradient_rate[1][1] > deformation_gradient_rate[2][2]);
                    Ok(())
                }
                #[test]
                fn minimize_biaxial_undeformed_inner() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &$constitutive_model_constructed.minimize_biaxial_inner(
                            &DeformationGradient::identity(),
                            0.0,
                            0.0,
                            &$solver,
                            &DeformationGradientRate::zero(),
                        )?,
                        &DeformationGradientRate::zero(),
                    )
                }
                #[test]
                fn root_uniaxial_compression_inner() -> Result<(), TestError> {
                    let deformation_gradient_rate = $constitutive_model_constructed
                        .root_uniaxial_inner(
                            &DeformationGradient::identity(),
                            -4.4,
                            &$solver,
                            &DeformationGradientRate::zero(),
                        )?;
                    assert!(deformation_gradient_rate.is_diagonal());
                    assert!(deformation_gradient_rate[0][0] < deformation_gradient_rate[1][1]);
                    assert_eq_within_tols(
                        &deformation_gradient_rate[1][1],
                        &deformation_gradient_rate[2][2],
                    )
                }
                #[test]
                fn root_uniaxial_tension_inner() -> Result<(), TestError> {
                    let deformation_gradient_rate = $constitutive_model_constructed
                        .root_uniaxial_inner(
                            &DeformationGradient::identity(),
                            4.4,
                            &$solver,
                            &DeformationGradientRate::zero(),
                        )?;
                    assert!(deformation_gradient_rate.is_diagonal());
                    assert!(deformation_gradient_rate[0][0] > deformation_gradient_rate[1][1]);
                    assert_eq_within_tols(
                        &deformation_gradient_rate[1][1],
                        &deformation_gradient_rate[2][2],
                    )
                }
                #[test]
                fn root_uniaxial_undeformed_inner() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &$constitutive_model_constructed.root_uniaxial_inner(
                            &DeformationGradient::identity(),
                            0.0,
                            &$solver,
                            &DeformationGradientRate::zero(),
                        )?,
                        &DeformationGradientRate::zero(),
                    )
                }
                #[test]
                fn root_biaxial_compression_inner() -> Result<(), TestError> {
                    let deformation_gradient_rate = $constitutive_model_constructed
                        .root_biaxial_inner(
                            &DeformationGradient::identity(),
                            -0.77,
                            -0.88,
                            &$solver,
                            &DeformationGradientRate::zero(),
                        )?;
                    assert!(deformation_gradient_rate.is_diagonal());
                    assert!(deformation_gradient_rate[0][0] < deformation_gradient_rate[2][2]);
                    assert!(deformation_gradient_rate[1][1] < deformation_gradient_rate[2][2]);
                    Ok(())
                }
                #[test]
                fn root_biaxial_mixed_inner() -> Result<(), TestError> {
                    let deformation_gradient_rate = $constitutive_model_constructed
                        .root_biaxial_inner(
                            &DeformationGradient::identity(),
                            1.3,
                            -0.64,
                            &$solver,
                            &DeformationGradientRate::zero(),
                        )?;
                    assert!(deformation_gradient_rate.is_diagonal());
                    assert!(deformation_gradient_rate[0][0] > deformation_gradient_rate[2][2]);
                    assert!(deformation_gradient_rate[1][1] < deformation_gradient_rate[2][2]);
                    Ok(())
                }
                #[test]
                fn root_biaxial_tension_inner() -> Result<(), TestError> {
                    let deformation_gradient_rate = $constitutive_model_constructed
                        .root_biaxial_inner(
                            &DeformationGradient::identity(),
                            1.3,
                            1.2,
                            &$solver,
                            &DeformationGradientRate::zero(),
                        )?;
                    assert!(deformation_gradient_rate.is_diagonal());
                    assert!(deformation_gradient_rate[0][0] > deformation_gradient_rate[2][2]);
                    assert!(deformation_gradient_rate[1][1] > deformation_gradient_rate[2][2]);
                    Ok(())
                }
                #[test]
                fn root_biaxial_undeformed_inner() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &$constitutive_model_constructed.root_biaxial_inner(
                            &DeformationGradient::identity(),
                            0.0,
                            0.0,
                            &$solver,
                            &DeformationGradientRate::zero(),
                        )?,
                        &DeformationGradientRate::zero(),
                    )
                }
            };
        }
        // mod gradient_descent_root_dual {
        //     use super::*;
        //     use crate::{
        //         math::optimize::GradientDescent,
        //     };
        //     test_with_solver!(GradientDescent {
        //         dual: true,
        //         ..Default::default()
        //     });
        // }
        mod newton_raphson {
            use super::*;
            use crate::math::optimize::NewtonRaphson;
            test_with_solver!(NewtonRaphson::default());
        }
    };
}
pub(crate) use test_minimize_and_root;

macro_rules! viscous_dissipation_from_deformation_gradient_rate_simple {
    ($constitutive_model_constructed: expr, $deformation_gradient_rate: expr) => {
        $constitutive_model_constructed
            .viscous_dissipation(&DeformationGradient::identity(), $deformation_gradient_rate)
    };
}
pub(crate) use viscous_dissipation_from_deformation_gradient_rate_simple;

macro_rules! viscous_dissipation_from_deformation_gradient_and_deformation_gradient_rate {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr, $deformation_gradient_rate: expr) => {
        $constitutive_model_constructed
            .viscous_dissipation($deformation_gradient, $deformation_gradient_rate)
    };
}
pub(crate) use viscous_dissipation_from_deformation_gradient_and_deformation_gradient_rate;

macro_rules! dissipation_potential_from_deformation_gradient_and_deformation_gradient_rate {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr, $deformation_gradient_rate: expr) => {
        $constitutive_model_constructed
            .dissipation_potential($deformation_gradient, $deformation_gradient_rate)
    };
}
pub(crate) use dissipation_potential_from_deformation_gradient_and_deformation_gradient_rate;

macro_rules! use_viscoelastic_macros
{
    () =>
    {
        use crate::constitutive::solid::viscoelastic::test::
        {
            cauchy_stress_from_deformation_gradient,
            cauchy_stress_from_deformation_gradient_simple,
            cauchy_stress_from_deformation_gradient_rotated,
            cauchy_stress_from_deformation_gradient_and_deformation_gradient_rate,
            cauchy_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate,
            first_piola_kirchhoff_stress_from_deformation_gradient,
            first_piola_kirchhoff_stress_from_deformation_gradient_simple,
            first_piola_kirchhoff_stress_from_deformation_gradient_rotated,
            first_piola_kirchhoff_stress_from_deformation_gradient_rate_simple,
            first_piola_kirchhoff_stress_from_deformation_gradient_and_deformation_gradient_rate,
            first_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate,
            second_piola_kirchhoff_stress_from_deformation_gradient,
            second_piola_kirchhoff_stress_from_deformation_gradient_simple,
            second_piola_kirchhoff_stress_from_deformation_gradient_rotated,
            second_piola_kirchhoff_stress_from_deformation_gradient_and_deformation_gradient_rate,
            second_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate,
        };
    }
}
pub(crate) use use_viscoelastic_macros;

macro_rules! test_solid_elastic_hyperviscous_constitutive_model
{
    ($constitutive_model: ident, $constitutive_model_parameters: expr, $constitutive_model_constructed: expr) =>
    {
        crate::constitutive::solid::elastic::test::test_solid_constitutive_construction!(
            $constitutive_model, $constitutive_model_parameters, $constitutive_model_constructed
        );
        crate::constitutive::solid::elastic::test::test_solid_constitutive_model_no_tangents!(
            $constitutive_model_constructed
        );
        crate::constitutive::solid::viscoelastic::test::test_solid_viscous_constitutive_model!(
            $constitutive_model, $constitutive_model_parameters, $constitutive_model_constructed
        );
        crate::constitutive::solid::elastic_hyperviscous::test::test_solid_elastic_hyperviscous_specifics!(
            $constitutive_model, $constitutive_model_parameters, $constitutive_model_constructed
        );
    }
}
pub(crate) use test_solid_elastic_hyperviscous_constitutive_model;

macro_rules! test_solid_elastic_hyperviscous_specifics
{
    ($constitutive_model: ident, $constitutive_model_parameters: expr, $constitutive_model_constructed: expr) =>
    {
        mod elastic_hyperviscous
        {
            use super::*;
            mod viscous_dissipation // eventually should go in fluid/hyperviscous/test.rs
            {
                use super::*;
                fn first_piola_kirchhoff_stress_from_finite_difference_of_viscous_dissipation(is_deformed: bool) ->  Result<FirstPiolaKirchhoffStress, TestError>
                {
                    let mut first_piola_kirchhoff_stress = FirstPiolaKirchhoffStress::zero();
                    for i in 0..3
                    {
                        for j in 0..3
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
                            deformation_gradient_rate_plus[i][j] += 0.5*EPSILON;
                            let helmholtz_free_energy_density_plus =
                            viscous_dissipation_from_deformation_gradient_rate_simple!(
                                $constitutive_model_constructed, &deformation_gradient_rate_plus
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
                            deformation_gradient_rate_minus[i][j] -= 0.5*EPSILON;
                            let helmholtz_free_energy_density_minus =
                            viscous_dissipation_from_deformation_gradient_rate_simple!(
                                $constitutive_model_constructed, &deformation_gradient_rate_minus
                            )?;
                            first_piola_kirchhoff_stress[i][j] = (
                                helmholtz_free_energy_density_plus - helmholtz_free_energy_density_minus
                            )/EPSILON;
                        }
                    }
                    Ok(first_piola_kirchhoff_stress)
                }
                mod deformed
                {
                    use super::*;
                    #[test]
                    fn finite_difference() -> Result<(), TestError>
                    {
                        assert_eq_from_fd(
                            &first_piola_kirchhoff_stress_from_deformation_gradient_rate_simple!(
                                $constitutive_model_constructed, &get_deformation_gradient_rate()
                            )?,
                            &first_piola_kirchhoff_stress_from_finite_difference_of_viscous_dissipation(true)?
                        )
                    }
                    #[test]
                    fn minimized() -> Result<(), TestError>
                    {
                        let first_piola_kirchhoff_stress =
                        first_piola_kirchhoff_stress_from_deformation_gradient_rate_simple!(
                            $constitutive_model_constructed, &get_deformation_gradient_rate()
                        )?;
                        let minimum =
                        viscous_dissipation_from_deformation_gradient_rate_simple!(
                            $constitutive_model_constructed, &get_deformation_gradient_rate()
                        )? - first_piola_kirchhoff_stress.full_contraction(
                            &get_deformation_gradient_rate()
                        );
                        let mut perturbed_deformation_gradient_rate = get_deformation_gradient_rate();
                        (0..3).try_for_each(|i|
                            (0..3).try_for_each(|j|{
                                perturbed_deformation_gradient_rate = get_deformation_gradient_rate();
                                perturbed_deformation_gradient_rate[i][j] += 0.5 * EPSILON;
                                assert!(
                                    viscous_dissipation_from_deformation_gradient_rate_simple!(
                                        $constitutive_model_constructed, &perturbed_deformation_gradient_rate
                                    )? - first_piola_kirchhoff_stress.full_contraction(
                                        &perturbed_deformation_gradient_rate
                                    ) > minimum
                                );
                                perturbed_deformation_gradient_rate[i][j] -= EPSILON;
                                assert!(
                                    viscous_dissipation_from_deformation_gradient_rate_simple!(
                                        $constitutive_model_constructed, &perturbed_deformation_gradient_rate
                                    )? - first_piola_kirchhoff_stress.full_contraction(
                                        &perturbed_deformation_gradient_rate
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
                            &viscous_dissipation_from_deformation_gradient_and_deformation_gradient_rate!(
                                $constitutive_model_constructed, &get_deformation_gradient(), &get_deformation_gradient_rate()
                            )?,
                            &viscous_dissipation_from_deformation_gradient_and_deformation_gradient_rate!(
                                $constitutive_model_constructed, &get_deformation_gradient_rotated(), &get_deformation_gradient_rate_rotated()
                            )?
                        )
                    }
                    #[test]
                    fn positive() -> Result<(), TestError>
                    {
                        assert!(
                            viscous_dissipation_from_deformation_gradient_rate_simple!(
                                $constitutive_model_constructed,  &get_deformation_gradient_rate()
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
                            &first_piola_kirchhoff_stress_from_finite_difference_of_viscous_dissipation(false)?,
                            &FirstPiolaKirchhoffStress::zero()
                        )
                    }
                    #[test]
                    fn minimized() -> Result<(), TestError>
                    {
                        let minimum =
                        viscous_dissipation_from_deformation_gradient_rate_simple!(
                            $constitutive_model_constructed, &DeformationGradientRate::zero()
                        )?;
                        let mut perturbed_deformation_gradient_rate = DeformationGradientRate::zero();
                        (0..3).try_for_each(|i|
                            (0..3).try_for_each(|j|{
                                perturbed_deformation_gradient_rate = DeformationGradientRate::zero();
                                perturbed_deformation_gradient_rate[i][j] += 0.5 * EPSILON;
                                assert!(
                                    viscous_dissipation_from_deformation_gradient_rate_simple!(
                                        $constitutive_model_constructed, &perturbed_deformation_gradient_rate
                                    )? > minimum
                                );
                                perturbed_deformation_gradient_rate[i][j] -= EPSILON;
                                assert!(
                                    viscous_dissipation_from_deformation_gradient_rate_simple!(
                                        $constitutive_model_constructed, &perturbed_deformation_gradient_rate
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
                            &viscous_dissipation_from_deformation_gradient_rate_simple!(
                                $constitutive_model_constructed,  &DeformationGradientRate::zero()
                            )?, &0.0
                        )
                    }
                }
            }
            mod dissipation_potential
            {
                use super::*;
                fn first_piola_kirchhoff_stress_from_finite_difference_of_dissipation_potential(is_deformed: bool) -> Result<FirstPiolaKirchhoffStress, TestError>
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
                    let mut first_piola_kirchhoff_stress = FirstPiolaKirchhoffStress::zero();
                    for i in 0..3
                    {
                        for j in 0..3
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
                            deformation_gradient_rate_plus[i][j] += 0.5*EPSILON;
                            let helmholtz_free_energy_density_plus =
                            dissipation_potential_from_deformation_gradient_and_deformation_gradient_rate!(
                                $constitutive_model_constructed, &deformation_gradient, &deformation_gradient_rate_plus
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
                            deformation_gradient_rate_minus[i][j] -= 0.5*EPSILON;
                            let helmholtz_free_energy_density_minus =
                            dissipation_potential_from_deformation_gradient_and_deformation_gradient_rate!(
                                $constitutive_model_constructed, &deformation_gradient, &deformation_gradient_rate_minus
                            )?;
                            first_piola_kirchhoff_stress[i][j] = (
                                helmholtz_free_energy_density_plus - helmholtz_free_energy_density_minus
                            )/EPSILON;
                        }
                    }
                    Ok(first_piola_kirchhoff_stress)
                }
                mod deformed
                {
                    use super::*;
                    #[test]
                    fn finite_difference() -> Result<(), TestError>
                    {
                        assert_eq_from_fd(
                            &first_piola_kirchhoff_stress_from_deformation_gradient_and_deformation_gradient_rate!(
                                $constitutive_model_constructed, &get_deformation_gradient(), &get_deformation_gradient_rate()
                            )?,
                            &first_piola_kirchhoff_stress_from_finite_difference_of_dissipation_potential(true)?
                        )
                    }
                    #[test]
                    fn minimized() -> Result<(), TestError>
                    {
                        let first_piola_kirchhoff_stress =
                        first_piola_kirchhoff_stress_from_deformation_gradient_and_deformation_gradient_rate!(
                            $constitutive_model_constructed, &get_deformation_gradient(), &get_deformation_gradient_rate()
                        )?;
                        let minimum =
                        dissipation_potential_from_deformation_gradient_and_deformation_gradient_rate!(
                            $constitutive_model_constructed, &get_deformation_gradient(), &get_deformation_gradient_rate()
                        )? - first_piola_kirchhoff_stress.full_contraction(
                            &get_deformation_gradient_rate()
                        );
                        let mut perturbed_deformation_gradient_rate = get_deformation_gradient_rate();
                        (0..3).try_for_each(|i|
                            (0..3).try_for_each(|j|{
                                perturbed_deformation_gradient_rate = get_deformation_gradient_rate();
                                perturbed_deformation_gradient_rate[i][j] += 0.5 * EPSILON;
                                assert!(
                                    dissipation_potential_from_deformation_gradient_and_deformation_gradient_rate!(
                                        $constitutive_model_constructed, &get_deformation_gradient(), &perturbed_deformation_gradient_rate
                                    )? - first_piola_kirchhoff_stress.full_contraction(
                                        &perturbed_deformation_gradient_rate
                                    ) > minimum
                                );
                                perturbed_deformation_gradient_rate[i][j] -= EPSILON;
                                assert!(
                                    dissipation_potential_from_deformation_gradient_and_deformation_gradient_rate!(
                                        $constitutive_model_constructed, &get_deformation_gradient(), &perturbed_deformation_gradient_rate
                                    )? - first_piola_kirchhoff_stress.full_contraction(
                                        &perturbed_deformation_gradient_rate
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
                            &dissipation_potential_from_deformation_gradient_and_deformation_gradient_rate!(
                                $constitutive_model_constructed, &get_deformation_gradient(), &get_deformation_gradient_rate()
                            )?,
                            &dissipation_potential_from_deformation_gradient_and_deformation_gradient_rate!(
                                $constitutive_model_constructed, &get_deformation_gradient_rotated(), &get_deformation_gradient_rate_rotated()
                            )?
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
                            &first_piola_kirchhoff_stress_from_finite_difference_of_dissipation_potential(false)?,
                            &FirstPiolaKirchhoffStress::zero()
                        )
                    }
                    #[test]
                    fn minimized() -> Result<(), TestError>
                    {
                        let minimum =
                        dissipation_potential_from_deformation_gradient_and_deformation_gradient_rate!(
                            $constitutive_model_constructed, &DeformationGradient::identity(), &DeformationGradientRate::zero()
                        )?;
                        let mut perturbed_deformation_gradient_rate = DeformationGradientRate::zero();
                        (0..3).try_for_each(|i|
                            (0..3).try_for_each(|j|{
                                perturbed_deformation_gradient_rate = DeformationGradientRate::zero();
                                perturbed_deformation_gradient_rate[i][j] += 0.5 * EPSILON;
                                assert!(
                                    dissipation_potential_from_deformation_gradient_and_deformation_gradient_rate!(
                                        $constitutive_model_constructed, &DeformationGradient::identity(), &perturbed_deformation_gradient_rate
                                    )? > minimum
                                );
                                perturbed_deformation_gradient_rate[i][j] -= EPSILON;
                                assert!(
                                    dissipation_potential_from_deformation_gradient_and_deformation_gradient_rate!(
                                        $constitutive_model_constructed, &DeformationGradient::identity(), &perturbed_deformation_gradient_rate
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
                            &dissipation_potential_from_deformation_gradient_and_deformation_gradient_rate!(
                                $constitutive_model_constructed, &DeformationGradient::identity(), &DeformationGradientRate::zero()
                            )?, &0.0
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
                    fn symmetry() -> Result<(), TestError>
                    {
                        let first_piola_kirchhoff_rate_tangent_stiffness =
                        first_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                            $constitutive_model_constructed, &get_deformation_gradient(), &get_deformation_gradient_rate()
                        )?;
                        assert_eq_within_tols(
                            &first_piola_kirchhoff_rate_tangent_stiffness,
                            &(0..3).map(|i|
                                (0..3).map(|j|
                                    (0..3).map(|k|
                                        (0..3).map(|l|
                                            first_piola_kirchhoff_rate_tangent_stiffness[k][l][i][j].clone()
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
                        let first_piola_kirchhoff_rate_tangent_stiffness =
                        first_piola_kirchhoff_rate_tangent_stiffness_from_deformation_gradient_and_deformation_gradient_rate!(
                            $constitutive_model_constructed, &DeformationGradient::identity(), &DeformationGradientRate::zero()
                        )?;
                        assert_eq_within_tols(
                            &first_piola_kirchhoff_rate_tangent_stiffness,
                            &(0..3).map(|i|
                                (0..3).map(|j|
                                    (0..3).map(|k|
                                        (0..3).map(|l|
                                            first_piola_kirchhoff_rate_tangent_stiffness[k][l][i][j].clone()
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
pub(crate) use test_solid_elastic_hyperviscous_specifics;
