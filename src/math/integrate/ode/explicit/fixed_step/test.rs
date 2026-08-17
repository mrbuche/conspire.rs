macro_rules! test_explicit_fixed_step {
    ($integration: expr) => {
        use crate::math::{
            Scalar, Tensor,
            integrate::{
                FixedStep,
                ode::explicit::test::test_explicit,
                test::{LENGTH, zero_to_one},
            },
        };
        const TIME_STEP: Quantity<Time> = Time::seconds(0.1);
        const TOLERANCE: Scalar = 0.1;
        test_explicit!($integration);
        #[test]
        fn dxdt_eq_neg_x() -> Result<(), AssertionError> {
            $crate::math::assert::Assert::eq(&FixedStep::<Time>::dt(&$integration), &TIME_STEP)?;
            let (time, solution, function): (
                Times,
                TensorVector<Quantity>,
                TensorVector<Quantity<Rate>>,
            ) = $integration.integrate(
                |_: Quantity<Time>, x: &Quantity| Ok(x * -RATE),
                &[Quantity::new(0.0), Quantity::new(0.8)],
                Quantity::new(1.0),
            )?;
            time.iter()
                .zip(solution.iter().zip(function.iter()))
                .try_for_each(|(t, (y, f))| {
                    $crate::math::assert::Assert {
                        abs_tol: TOLERANCE,
                        rel_tol: TOLERANCE,
                        ..Default::default()
                    }
                    .eq_within_tols(y, &(-(*t * RATE)).exp())?;
                    $crate::math::assert::Assert {
                        abs_tol: TOLERANCE,
                        rel_tol: TOLERANCE,
                        ..Default::default()
                    }
                    .eq_within_tols(f, &(y * -RATE))
                })
        }
        #[test]
        fn eval_times() -> Result<(), AssertionError> {
            $crate::math::assert::Assert::eq(&FixedStep::<Time>::dt(&$integration), &TIME_STEP)?;
            let (time, solution, function): (
                Times,
                TensorVector<Quantity>,
                TensorVector<Quantity<Rate>>,
            ) = $integration.integrate(
                |_: Quantity<Time>, x: &Quantity| Ok(x * -RATE),
                &zero_to_one::<LENGTH>(),
                Quantity::new(1.0),
            )?;
            time.iter()
                .zip(solution.iter().zip(function.iter()))
                .try_for_each(|(t, (y, f))| {
                    $crate::math::assert::Assert {
                        abs_tol: TOLERANCE,
                        rel_tol: TOLERANCE,
                        ..Default::default()
                    }
                    .eq_within_tols(y, &(-(*t * RATE)).exp())?;
                    $crate::math::assert::Assert {
                        abs_tol: TOLERANCE,
                        rel_tol: TOLERANCE,
                        ..Default::default()
                    }
                    .eq_within_tols(f, &(y * -RATE))
                })
        }
    };
}
pub(crate) use test_explicit_fixed_step;
