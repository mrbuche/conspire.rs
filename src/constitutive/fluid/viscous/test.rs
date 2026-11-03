macro_rules! finite_difference_tangent {
    ($name:ident, $stress:ident, $tangent:ident, $type:path) => {
        #[test]
        fn $name() -> Result<(), $crate::math::assert::AssertionError> {
            use $crate::math::{
                TensorArray as _,
                assert::{Assert, FiniteDifference as _, perturbation},
            };
            let deformation_gradient = deformation_gradient();
            let deformation_gradient_rate = deformation_gradient_rate();
            let model = model();
            let tangent = model.$tangent(&deformation_gradient, &deformation_gradient_rate)?;
            let mut fd = <$type>::zero();
            for k in 0..3 {
                for l in 0..3 {
                    let mut plus = deformation_gradient.clone();
                    plus[k][l] += perturbation(0.5 * $crate::EPSILON);
                    let stress_plus = model.$stress(&plus, &deformation_gradient_rate)?;
                    let mut minus = deformation_gradient.clone();
                    minus[k][l] -= perturbation(0.5 * $crate::EPSILON);
                    let stress_minus = model.$stress(&minus, &deformation_gradient_rate)?;
                    for i in 0..3 {
                        for j in 0..3 {
                            fd[i][j][k][l] = (stress_plus[i][j] - stress_minus[i][j])
                                / perturbation::<$crate::units::Dimensionless>($crate::EPSILON);
                        }
                    }
                }
            }
            if tangent.error_fd(&fd, 5e1 * $crate::EPSILON).is_some() {
                Assert::default().eq_within_fd_tol(&tangent, &fd)
            } else {
                Ok(())
            }
        }
    };
}

pub(crate) use finite_difference_tangent;
