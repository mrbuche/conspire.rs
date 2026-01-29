macro_rules! test_implicit_fixed_step {
    ($integration: expr) => {
        use crate::{
            EPSILON,
            math::{
                Scalar, Tensor, Vector, assert_eq,
                integrate::{FixedStep, IntegrationError},
                test::TestError,
                test::assert_eq_from_fd,
            },
        };
        const TIME_STEP: Scalar = 0.1;
        const TOLERANCE: Scalar = TIME_STEP;
        #[test]
        fn finite_difference() -> Result<(), TestError> {
            use crate::math::integrate::{ImplicitFirstOrder, ImplicitZerothOrder};
            let t = 0.55_f64;
            let y = t.sin();
            let function =
                |_t: Scalar, y: &Scalar| Ok::<Scalar, IntegrationError>(-0.5 * t * y.powi(2));
            let jacobian = |_t: Scalar, y: &Scalar| Ok::<Scalar, IntegrationError>(-t * y);
            let dt = 0.1;
            let t_trial = t + dt;
            let y_trial = y + function(t, &y)? * dt;
            let finite_difference = (ImplicitZerothOrder::<Scalar, Vector>::residual(
                &$integration,
                &function,
                t,
                &y,
                t_trial,
                &(y_trial + 0.5 * EPSILON),
                dt,
            )? - ImplicitZerothOrder::<Scalar, Vector>::residual(
                &$integration,
                &function,
                t,
                &y,
                t_trial,
                &(y_trial - 0.5 * EPSILON),
                dt,
            )?) / EPSILON;
            assert_eq_from_fd(
                &ImplicitFirstOrder::<Scalar, Scalar, Vector>::hessian(
                    &$integration,
                    &jacobian,
                    t,
                    &y,
                    t_trial,
                    &y_trial,
                    dt,
                )?,
                &finite_difference,
            )
        }
        mod gradient_descent {
            use super::*;
            use crate::math::{integrate::ImplicitZerothOrder, optimize::GradientDescent};
            #[test]
            fn first_order_tensor_rank_0() -> Result<(), TestError> {
                assert_eq(&$integration.dt(), &TIME_STEP)?;
                let (time, solution, function): (Vector, Vector, _) = $integration.integrate(
                    |t: Scalar, _: &Scalar| Ok(t),
                    &[0.0, 1.0],
                    0.0,
                    GradientDescent::default(),
                )?;
                time.iter()
                    .zip(solution.iter().zip(function.iter()))
                    .for_each(|(t, (y, f))| {
                        assert!((0.5 * t * t - y).abs() < TOLERANCE && (t - f).abs() < TOLERANCE)
                    });
                Ok(())
            }
        }
        mod newton_raphson {
            use super::*;
            use crate::math::{integrate::ImplicitFirstOrder, optimize::NewtonRaphson};
            #[test]
            fn first_order_tensor_rank_0() -> Result<(), TestError> {
                assert_eq(&$integration.dt(), &TIME_STEP)?;
                let (time, solution, function): (Vector, Vector, _) = $integration.integrate(
                    |t: Scalar, _: &Scalar| Ok(t),
                    |_: Scalar, _: &Scalar| Ok(1.0),
                    &[0.0, 1.0],
                    0.0,
                    NewtonRaphson::default(),
                )?;
                time.iter()
                    .zip(solution.iter().zip(function.iter()))
                    .for_each(|(t, (y, f))| {
                        assert!((0.5 * t * t - y).abs() < TOLERANCE && (t - f).abs() < TOLERANCE)
                    });
                Ok(())
            }
        }
    };
}
pub(crate) use test_implicit_fixed_step;
