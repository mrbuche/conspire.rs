use crate::constitutive::hybrid::{
    Multiplicative, elastic::test::test_hybrid_elastic_constitutive_models_no_tangents,
};

test_hybrid_elastic_constitutive_models_no_tangents!(Multiplicative);

use crate::{
    constitutive::solid::elastic::internal_variables::ElasticIV,
    math::test::{ErrorTensor, assert_eq_from_fd},
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
