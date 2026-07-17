macro_rules! test_explicit_fixed_step {
    ($integration: expr) => {
        use crate::math::{
            Tensor,
            integrate::{
                FixedStep,
                ode::explicit::test::test_explicit,
                test::{LENGTH, zero_to_one},
            },
        };
        const TIME_STEP: Scalar = 0.1;
        const TOLERANCE: Scalar = TIME_STEP;
        test_explicit!($integration);
        #[test]
        fn dxdt_eq_neg_x() -> Result<(), AssertionError> {
            $crate::math::assert::Assert::eq(&$integration.dt(), &TIME_STEP)?;
            let (time, solution, function): (Vector, Vector, _) =
                $integration.integrate(|_: Scalar, x: &Scalar| Ok(-x), &[0.0, 0.8], 1.0)?;
            time.iter()
                .zip(solution.iter().zip(function.iter()))
                .try_for_each(|(t, (y, f))| {
                    $crate::math::assert::Assert {
                        abs_tol: TOLERANCE,
                        rel_tol: TOLERANCE,
                        ..Default::default()
                    }
                    .eq_within_tols(y, &(-t).exp())?;
                    $crate::math::assert::Assert {
                        abs_tol: TOLERANCE,
                        rel_tol: TOLERANCE,
                        ..Default::default()
                    }
                    .eq_within_tols(f, &-y)
                })
        }
        #[test]
        fn eval_times() -> Result<(), AssertionError> {
            $crate::math::assert::Assert::eq(&$integration.dt(), &TIME_STEP)?;
            let (time, solution, function): (Vector, Vector, _) = $integration.integrate(
                |_: Scalar, x: &Scalar| Ok(-x),
                &zero_to_one::<LENGTH>(),
                1.0,
            )?;
            time.iter()
                .zip(solution.iter().zip(function.iter()))
                .try_for_each(|(t, (y, f))| {
                    $crate::math::assert::Assert {
                        abs_tol: TOLERANCE,
                        rel_tol: TOLERANCE,
                        ..Default::default()
                    }
                    .eq_within_tols(y, &(-t).exp())?;
                    $crate::math::assert::Assert {
                        abs_tol: TOLERANCE,
                        rel_tol: TOLERANCE,
                        ..Default::default()
                    }
                    .eq_within_tols(f, &-y)
                })
        }
    };
}
pub(crate) use test_explicit_fixed_step;
