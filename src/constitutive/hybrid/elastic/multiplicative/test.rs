use crate::constitutive::hybrid::{
    Multiplicative, elastic::test::test_hybrid_elastic_constitutive_models_no_tangents,
};

test_hybrid_elastic_constitutive_models_no_tangents!(Multiplicative);

use crate::{
    constitutive::solid::elastic::{AppliedLoad, internal_variables::ElasticIV},
    math::{
        TensorRank4,
        optimize::{GradientDescent, NewtonRaphson},
        test::{ErrorTensor, assert_eq_from_fd},
    },
    mechanics::*,
};

#[test]
fn finite_difference_foo_0() -> Result<(), TestError> {
    let deformation_gradient = DeformationGradient::new([
        [1.31924942, 1.36431217, 0.41764434],
        [0.09959341, 1.38409741, 1.48320137],
        [0.21114106, 1.16675104, 1.98146028],
    ]);
    let deformation_gradient_2 = DeformationGradient2::new([
        [0.84598947, 1.44803635, 0.62447529],
        [0.76208429, 1.94584131, 0.74035917],
        [1.93680854, 2.32953025, 3.36786684],
    ]);
    let model = Multiplicative::from((
        AlmansiHamel {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
    ));
    let tangent =
        model.cauchy_tangent_stiffness_foo(&deformation_gradient, &deformation_gradient_2)?;
    let mut fd = CauchyTangentStiffness::zero();
    for k in 0..3 {
        for l in 0..3 {
            let mut deformation_gradient_plus = deformation_gradient.clone();
            deformation_gradient_plus[k][l] += 0.5 * crate::EPSILON;
            let cauchy_stress_plus =
                model.cauchy_stress_foo(&deformation_gradient_plus, &deformation_gradient_2)?;
            let mut deformation_gradient_minus = deformation_gradient.clone();
            deformation_gradient_minus[k][l] -= 0.5 * crate::EPSILON;
            let cauchy_stress_minus =
                model.cauchy_stress_foo(&deformation_gradient_minus, &deformation_gradient_2)?;
            for i in 0..3 {
                for j in 0..3 {
                    fd[i][j][k][l] =
                        (cauchy_stress_plus[i][j] - cauchy_stress_minus[i][j]) / crate::EPSILON;
                }
            }
        }
    }
    if tangent.error_fd(&fd, 5e1 * crate::EPSILON).is_some() {
        assert_eq_from_fd(&tangent, &fd)
    } else {
        Ok(())
    }
}

#[test]
fn finite_difference_foo_1() -> Result<(), TestError> {
    let deformation_gradient = DeformationGradient::new([
        [1.31924942, 1.36431217, 0.41764434],
        [0.09959341, 1.38409741, 1.48320137],
        [0.21114106, 1.16675104, 1.98146028],
    ]);
    let deformation_gradient_2 = DeformationGradient2::new([
        [0.84598947, 1.44803635, 0.62447529],
        [0.76208429, 1.94584131, 0.74035917],
        [1.93680854, 2.32953025, 3.36786684],
    ]);
    let model = Multiplicative::from((
        AlmansiHamel {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
    ));
    let (tangent_1, _, _) =
        model.internal_variables_tangents(&deformation_gradient, &deformation_gradient_2)?;
    let mut fd = TensorRank4::zero();
    for k in 0..3 {
        for l in 0..3 {
            let mut deformation_gradient_2_plus = deformation_gradient_2.clone();
            deformation_gradient_2_plus[k][l] += 0.5 * crate::EPSILON;
            let residual_plus = model.first_piola_kirchhoff_stress_foo(
                &deformation_gradient,
                &deformation_gradient_2_plus,
            )?;
            let mut deformation_gradient_2_minus = deformation_gradient_2.clone();
            deformation_gradient_2_minus[k][l] -= 0.5 * crate::EPSILON;
            let residual_minus = model.first_piola_kirchhoff_stress_foo(
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
    if tangent_1.error_fd(&fd, 5e1 * crate::EPSILON).is_some() {
        assert_eq_from_fd(&tangent_1, &fd)
    } else {
        Ok(())
    }
}

#[test]
fn finite_difference_foo_2() -> Result<(), TestError> {
    let deformation_gradient = DeformationGradient::new([
        [1.31924942, 1.36431217, 0.41764434],
        [0.09959341, 1.38409741, 1.48320137],
        [0.21114106, 1.16675104, 1.98146028],
    ]);
    let deformation_gradient_2 = DeformationGradient2::new([
        [0.84598947, 1.44803635, 0.62447529],
        [0.76208429, 1.94584131, 0.74035917],
        [1.93680854, 2.32953025, 3.36786684],
    ]);
    let model = Multiplicative::from((
        AlmansiHamel {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
    ));
    let (_, tangent_2, _) =
        model.internal_variables_tangents(&deformation_gradient, &deformation_gradient_2)?;
    let mut fd = TensorRank4::zero();
    for k in 0..3 {
        for l in 0..3 {
            let mut deformation_gradient_plus = deformation_gradient.clone();
            deformation_gradient_plus[k][l] += 0.5 * crate::EPSILON;
            let residual_plus = model
                .internal_variables_residual(&deformation_gradient_plus, &deformation_gradient_2)?;
            let mut deformation_gradient_minus = deformation_gradient.clone();
            deformation_gradient_minus[k][l] -= 0.5 * crate::EPSILON;
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
    if tangent_2.error_fd(&fd, 5e1 * crate::EPSILON).is_some() {
        assert_eq_from_fd(&tangent_2, &fd)
    } else {
        Ok(())
    }
}

#[test]
fn finite_difference_foo_3() -> Result<(), TestError> {
    let deformation_gradient = DeformationGradient::new([
        [1.31924942, 1.36431217, 0.41764434],
        [0.09959341, 1.38409741, 1.48320137],
        [0.21114106, 1.16675104, 1.98146028],
    ]);
    let deformation_gradient_2 = DeformationGradient2::new([
        [0.84598947, 1.44803635, 0.62447529],
        [0.76208429, 1.94584131, 0.74035917],
        [1.93680854, 2.32953025, 3.36786684],
    ]);
    let model = Multiplicative::from((
        AlmansiHamel {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
    ));
    let (_, _, tangent_3) =
        model.internal_variables_tangents(&deformation_gradient, &deformation_gradient_2)?;
    let mut fd = TensorRank4::zero();
    for k in 0..3 {
        for l in 0..3 {
            let mut deformation_gradient_2_plus = deformation_gradient_2.clone();
            deformation_gradient_2_plus[k][l] += 0.5 * crate::EPSILON;
            let residual_plus = model
                .internal_variables_residual(&deformation_gradient, &deformation_gradient_2_plus)?;
            let mut deformation_gradient_2_minus = deformation_gradient_2.clone();
            deformation_gradient_2_minus[k][l] -= 0.5 * crate::EPSILON;
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
        assert_eq_from_fd(&tangent_3, &fd)
    } else {
        Ok(())
    }
}

const STRETCH: Scalar = 1.5;

#[test]
fn root_0() -> Result<(), TestError> {
    use crate::constitutive::solid::elastic::ZerothOrderRoot;
    let model = Multiplicative::from((
        AlmansiHamel {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
    ));
    let time = std::time::Instant::now();
    let _f = model.root(
        AppliedLoad::UniaxialStress(STRETCH),
        GradientDescent {
            dual: true,
            ..Default::default()
        },
    )?;
    println!("old_0 {:?}", time.elapsed());
    // let f_1 = &f * f_2.inverse();
    // println!("{}\n{}\n{}", f, f_1, f_2,);
    // println!("{}", f);
    Ok(())
}

#[test]
fn root_0_foo() -> Result<(), TestError> {
    use crate::constitutive::solid::elastic::internal_variables::ZerothOrderRoot;
    let model = Multiplicative::from((
        AlmansiHamel {
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

#[test]
fn root_1_foo() -> Result<(), TestError> {
    use crate::constitutive::solid::elastic::internal_variables::FirstOrderRoot;
    let model = Multiplicative::from((
        AlmansiHamel {
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
        NewtonRaphson::default(),
    )?;
    println!("new_1 {:?}", time.elapsed());
    // let _f_1 = &f * f_2.inverse();
    Ok(())
}
