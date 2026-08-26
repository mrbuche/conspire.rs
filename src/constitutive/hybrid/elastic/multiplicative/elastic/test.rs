use crate::math::assert::Assert;
use crate::math::assert::perturbation;
use crate::{
    constitutive::{
        hybrid::ElasticMultiplicative,
        solid::{
            elastic::{
                AlmansiHamelEulerian,
                test::{BULK_MODULUS, SHEAR_MODULUS},
            },
            hyperelastic::NeoHookean,
        },
    },
    math::{TensorArray, assert::AssertionError},
};

use crate::{
    constitutive::solid::elastic::{AppliedLoad, internal_variables::ElasticIV},
    math::{
        TensorRank4, Vector,
        assert::FiniteDifference,
        optimize::{GradientDescent, NewtonRaphson, SolveStrategy},
    },
    mechanics::*,
};

#[test]
fn finite_difference_0() -> Result<(), AssertionError> {
    let deformation_gradient = DeformationGradient::from([
        [1.31924942, 1.36431217, 0.41764434],
        [0.09959341, 1.38409741, 1.48320137],
        [0.21114106, 1.16675104, 1.98146028],
    ]);
    let deformation_gradient_2 = DeformationGradient2::from([
        [0.84598947, 1.44803635, 0.62447529],
        [0.76208429, 1.94584131, 0.74035917],
        [1.93680854, 2.32953025, 3.36786684],
    ]);
    let model = ElasticMultiplicative::from((
        AlmansiHamelEulerian {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
    ));
    let tangent = model.cauchy_tangent_stiffness(&deformation_gradient, &deformation_gradient_2)?;
    let mut fd = CauchyTangentStiffness::zero();
    for k in 0..3 {
        for l in 0..3 {
            let mut deformation_gradient_plus = deformation_gradient.clone();
            deformation_gradient_plus[k][l] += perturbation(0.5 * crate::EPSILON);
            let cauchy_stress_plus =
                model.cauchy_stress(&deformation_gradient_plus, &deformation_gradient_2)?;
            let mut deformation_gradient_minus = deformation_gradient.clone();
            deformation_gradient_minus[k][l] -= perturbation(0.5 * crate::EPSILON);
            let cauchy_stress_minus =
                model.cauchy_stress(&deformation_gradient_minus, &deformation_gradient_2)?;
            for i in 0..3 {
                for j in 0..3 {
                    fd[i][j][k][l] =
                        (cauchy_stress_plus[i][j] - cauchy_stress_minus[i][j]) / crate::EPSILON;
                }
            }
        }
    }
    if tangent.error_fd(&fd, 5e1 * crate::EPSILON).is_some() {
        Assert::default().eq_within_fd_tol(&tangent, &fd)
    } else {
        Ok(())
    }
}

#[test]
fn finite_difference_1() -> Result<(), AssertionError> {
    let deformation_gradient = DeformationGradient::from([
        [1.31924942, 1.36431217, 0.41764434],
        [0.09959341, 1.38409741, 1.48320137],
        [0.21114106, 1.16675104, 1.98146028],
    ]);
    let deformation_gradient_2 = DeformationGradient2::from([
        [0.84598947, 1.44803635, 0.62447529],
        [0.76208429, 1.94584131, 0.74035917],
        [1.93680854, 2.32953025, 3.36786684],
    ]);
    let model = ElasticMultiplicative::from((
        AlmansiHamelEulerian {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
    ));
    let (_, tangent_1, _, _) = model.tangents(&deformation_gradient, &deformation_gradient_2)?;
    let mut fd = TensorRank4::zero();
    for k in 0..3 {
        for l in 0..3 {
            let mut deformation_gradient_plus = deformation_gradient.clone();
            deformation_gradient_plus[k][l] += perturbation(0.5 * crate::EPSILON);
            let residual_plus = model
                .internal_variables_residual(&deformation_gradient_plus, &deformation_gradient_2)?;
            let mut deformation_gradient_minus = deformation_gradient.clone();
            deformation_gradient_minus[k][l] -= perturbation(0.5 * crate::EPSILON);
            let residual_minus = model.internal_variables_residual(
                &deformation_gradient_minus,
                &deformation_gradient_2,
            )?;
            for i in 0..3 {
                for j in 0..3 {
                    fd[i][j][k][l] = (residual_plus[i][j] - residual_minus[i][j]) / crate::EPSILON;
                }
            }
        }
    }
    if tangent_1.error_fd(&fd, 5e1 * crate::EPSILON).is_some() {
        Assert::default().eq_within_fd_tol(&tangent_1, &fd)
    } else {
        Ok(())
    }
}

#[test]
fn finite_difference_2() -> Result<(), AssertionError> {
    let deformation_gradient = DeformationGradient::from([
        [1.31924942, 1.36431217, 0.41764434],
        [0.09959341, 1.38409741, 1.48320137],
        [0.21114106, 1.16675104, 1.98146028],
    ]);
    let deformation_gradient_2 = DeformationGradient2::from([
        [0.84598947, 1.44803635, 0.62447529],
        [0.76208429, 1.94584131, 0.74035917],
        [1.93680854, 2.32953025, 3.36786684],
    ]);
    let model = ElasticMultiplicative::from((
        AlmansiHamelEulerian {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
    ));
    let (_, _, tangent_2, _) = model.tangents(&deformation_gradient, &deformation_gradient_2)?;
    let mut fd = TensorRank4::zero();
    for k in 0..3 {
        for l in 0..3 {
            let mut deformation_gradient_2_plus = deformation_gradient_2.clone();
            deformation_gradient_2_plus[k][l] += perturbation(0.5 * crate::EPSILON);
            let residual_plus = model.first_piola_kirchhoff_stress(
                &deformation_gradient,
                &deformation_gradient_2_plus,
            )?;
            let mut deformation_gradient_2_minus = deformation_gradient_2.clone();
            deformation_gradient_2_minus[k][l] -= perturbation(0.5 * crate::EPSILON);
            let residual_minus = model.first_piola_kirchhoff_stress(
                &deformation_gradient,
                &deformation_gradient_2_minus,
            )?;
            for i in 0..3 {
                for j in 0..3 {
                    fd[i][j][k][l] = (residual_plus[i][j] - residual_minus[i][j]) / crate::EPSILON;
                }
            }
        }
    }
    if tangent_2.error_fd(&fd, 5e1 * crate::EPSILON).is_some() {
        Assert::default().eq_within_fd_tol(&tangent_2, &fd)
    } else {
        Ok(())
    }
}

#[test]
fn finite_difference_3() -> Result<(), AssertionError> {
    let deformation_gradient = DeformationGradient::from([
        [1.31924942, 1.36431217, 0.41764434],
        [0.09959341, 1.38409741, 1.48320137],
        [0.21114106, 1.16675104, 1.98146028],
    ]);
    let deformation_gradient_2 = DeformationGradient2::from([
        [0.84598947, 1.44803635, 0.62447529],
        [0.76208429, 1.94584131, 0.74035917],
        [1.93680854, 2.32953025, 3.36786684],
    ]);
    let model = ElasticMultiplicative::from((
        AlmansiHamelEulerian {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
    ));
    let (_, _, _, tangent_3) = model.tangents(&deformation_gradient, &deformation_gradient_2)?;
    let mut fd = TensorRank4::zero();
    for k in 0..3 {
        for l in 0..3 {
            let mut deformation_gradient_2_plus = deformation_gradient_2.clone();
            deformation_gradient_2_plus[k][l] += perturbation(0.5 * crate::EPSILON);
            let residual_plus = model
                .internal_variables_residual(&deformation_gradient, &deformation_gradient_2_plus)?;
            let mut deformation_gradient_2_minus = deformation_gradient_2.clone();
            deformation_gradient_2_minus[k][l] -= perturbation(0.5 * crate::EPSILON);
            let residual_minus = model.internal_variables_residual(
                &deformation_gradient,
                &deformation_gradient_2_minus,
            )?;
            for i in 0..3 {
                for j in 0..3 {
                    fd[i][j][k][l] = (residual_plus[i][j] - residual_minus[i][j]) / crate::EPSILON;
                }
            }
        }
    }
    if tangent_3.error_fd(&fd, 5e1 * crate::EPSILON).is_some() {
        Assert::default().eq_within_fd_tol(&tangent_3, &fd)
    } else {
        Ok(())
    }
}

const STRETCH: Scalar = 1.5;

#[test]
fn root_0() -> Result<(), AssertionError> {
    use crate::constitutive::solid::elastic::internal_variables::ZerothOrderRoot;
    let model = ElasticMultiplicative::from((
        AlmansiHamelEulerian {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
    ));
    let time = std::time::Instant::now();
    let (_f, _f_2) = model.root(
        AppliedLoad::UniaxialStress(STRETCH),
        GradientDescent {
            dual: true,
            ..Default::default()
        },
    )?;
    println!("new_0 {:?}", time.elapsed());
    // let _f_1 = &f * f_2.inverse();
    Ok(())
}

fn model() -> ElasticMultiplicative<AlmansiHamelEulerian, NeoHookean> {
    ElasticMultiplicative::from((
        AlmansiHamelEulerian {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
    ))
}

fn rooted(
    strategy: SolveStrategy,
) -> Result<(DeformationGradient, DeformationGradient2), AssertionError> {
    use crate::constitutive::solid::elastic::internal_variables::FirstOrderRoot;
    let (f, f_2) = model().root(
        AppliedLoad::UniaxialStress(STRETCH),
        NewtonRaphson::default(),
        strategy,
    )?;
    Assert::default().zero_within_tols(&model().internal_variables_residual(&f, &f_2)?)?;
    Assert::default().eq_within_tols(
        Vector::from([
            f[0][0].value(),
            f[0][1].value(),
            f[0][2].value(),
            f[1][2].value(),
        ]),
        &Vector::from([STRETCH, 0.0, 0.0, 0.0]),
    )?;
    Ok((f, f_2))
}

#[test]
fn root_1() -> Result<(), AssertionError> {
    rooted(SolveStrategy::Monolithic { elimination: false })?;
    Ok(())
}

#[test]
fn root_1_elimination() -> Result<(), AssertionError> {
    let (f, f_2) = rooted(SolveStrategy::Monolithic { elimination: false })?;
    let (f_elim, f_2_elim) = rooted(SolveStrategy::Monolithic { elimination: true })?;
    Assert::default().eq_within_tols(&f_elim, &f)?;
    Assert::default().eq_within_tols(&f_2_elim, &f_2)
}

#[test]
fn root_1_condensed() -> Result<(), AssertionError> {
    let (f, f_2) = rooted(SolveStrategy::Monolithic { elimination: false })?;
    let (f_cond, f_2_cond) = rooted(SolveStrategy::Condensed(NewtonRaphson::default()))?;
    Assert::default().eq_within_tols(&f_cond, &f)?;
    Assert::default().eq_within_tols(&f_2_cond, &f_2)
}

#[test]
fn moduli() -> Result<(), AssertionError> {
    use crate::{
        EPSILON,
        constitutive::solid::Solid,
        math::{
            Rank2,
            optimize::{EqualityConstraint, FirstOrderRootFinding},
        },
    };
    let model = model();
    let fixed = model.internal_variables_fixed().to_vec();
    let solved = |f: &DeformationGradient| {
        NewtonRaphson::default().root(
            |v: &DeformationGradient2| Ok(model.internal_variables_residual(f, v)?),
            |v: &DeformationGradient2| Ok(model.tangents(f, v)?.3),
            model.internal_variables_initial(),
            EqualityConstraint::Fixed(fixed.clone()),
            None,
        )
    };
    let dilated = DeformationGradient::identity() * (1.0 + EPSILON / 3.0);
    let dilated_stress = model.first_piola_kirchhoff_stress(&dilated, &solved(&dilated)?)?;
    assert!(
        (3.0 * EPSILON * model.bulk_modulus().value() / dilated_stress.trace().value() - 1.0).abs()
            < EPSILON
    );
    let mut sheared = DeformationGradient::identity();
    sheared[0][1] = crate::math::Quantity::new(EPSILON);
    let sheared_stress = model.first_piola_kirchhoff_stress(&sheared, &solved(&sheared)?)?;
    assert!(
        (EPSILON * model.shear_modulus().value() / sheared_stress[0][1].value() - 1.0).abs()
            < EPSILON
    );
    Ok(())
}
