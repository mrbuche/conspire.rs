use crate::constitutive::hybrid::{
    Multiplicative, elastic::test::test_hybrid_elastic_constitutive_models_no_tangents,
};

test_hybrid_elastic_constitutive_models_no_tangents!(Multiplicative);

use crate::{
    constitutive::solid::elastic::{AppliedLoad, internal_variables::{ElasticIV, ZerothOrderRoot}},
    math::{
        optimize::{GradientDescent, NewtonRaphson},test::{ErrorTensor, assert_eq_from_fd}},
    mechanics::*,
};

#[test]
fn finite_difference_foo() -> Result<(), TestError> {
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
fn root_0_foo() -> Result<(), TestError> {
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
    let (f, f_2) = model.root(
        AppliedLoad::UniaxialStress(1.0),
        GradientDescent {
            dual: true,
            ..Default::default()
        },
    )?;
    // for (f_i, f_2_i) in t.iter().zip(f.iter().zip(f_p.iter())) {
    //     let (f_p_i, y_i) = s_i.into();
    //     let f_e = f_i * f_p_i.inverse();
    //     let c_e = model.cauchy_stress_foo(f_i, f_p_i)?;
    //     let m_e = f_e.transpose() * &c_e * f_e.inverse_transpose();
    //     let m_e_dev_mag = m_e.deviatoric().norm();
    //     println!(
    //         "[{}, {}, {}, {}, {}, {}, {}],",
    //         t_i,
    //         f_i[0][0],
    //         f_p_i[0][0],
    //         y_i,
    //         c_e[0][0],
    //         f_p_i.determinant(),
    //         m_e_dev_mag,
    //     )
    // }
    Ok(())
}
