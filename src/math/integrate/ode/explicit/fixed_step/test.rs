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
        #[test]
        fn derivative_is_evaluated_at_the_solution() -> Result<(), AssertionError> {
            let (time, solution, function): (
                Times,
                TensorVector<Quantity>,
                TensorVector<Quantity<Rate>>,
            ) = $integration.integrate(
                |_: Quantity<Time>, x: &Quantity| Ok(x * -RATE),
                &zero_to_one::<LENGTH>(),
                Quantity::new(1.0),
            )?;
            assert_eq!(time.len(), function.len());
            solution.iter().zip(function.iter()).try_for_each(|(y, f)| {
                $crate::math::assert::Assert {
                    abs_tol: 1e-12,
                    rel_tol: 1e-12,
                    ..Default::default()
                }
                .eq_within_tols(f, &(y * -RATE))
            })
        }
        #[test]
        fn function_evaluations_per_step() -> Result<(), AssertionError> {
            fn stages<I>(_: &I) -> usize
            where
                I: crate::math::integrate::FixedStepExplicit<
                        Quantity,
                        TensorVector<Quantity>,
                        TensorVector<Quantity<Rate>>,
                    >,
            {
                <I::Tableau as crate::math::integrate::ButcherTableau>::STAGES.min(
                    <I as crate::math::integrate::Explicit<
                        Quantity,
                        TensorVector<Quantity>,
                        TensorVector<Quantity<Rate>>,
                    >>::SLOPES,
                )
            }
            let mut evaluations = 0;
            let (time, ..): (Times, TensorVector<Quantity>, TensorVector<Quantity<Rate>>) =
                $integration.integrate(
                    |_: Quantity<Time>, x: &Quantity| {
                        evaluations += 1;
                        Ok(x * -RATE)
                    },
                    &zero_to_one::<LENGTH>(),
                    Quantity::new(1.0),
                )?;
            assert_eq!(evaluations, stages(&$integration) * (time.len() - 1) + 1);
            Ok(())
        }
    };
}
pub(crate) use test_explicit_fixed_step;
