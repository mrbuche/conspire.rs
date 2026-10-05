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
        fn stability<I>(_: &I) -> $crate::math::integrate::StabilityInterval
        where
            I: $crate::math::integrate::FixedStepExplicit<
                    Quantity,
                    TensorVector<Quantity>,
                    TensorVector<Quantity<Rate>>,
                >,
        {
            <I::Tableau as $crate::math::integrate::ButcherTableau>::stability()
        }
        fn bounded(
            spectrum: Spectrum,
            safety: Scalar,
        ) -> Result<(Times, TensorVector<Quantity>, TensorVector<Quantity<Rate>>), IntegrationError>
        {
            $integration.integrate_bounded(
                |_: Quantity<Time>, x: &Quantity| Ok(x * -RATE),
                |_, _| Ok(spectrum),
                safety,
                &[Quantity::new(0.0), Quantity::new(0.8)],
                Quantity::new(1.0),
            )
        }
        #[test]
        fn bounded_within_limit() {
            let scale = 2.0 * TIME_STEP.value() / stability(&$integration).real;
            assert!(bounded(Spectrum::Real(Time::seconds(scale)), 1.0).is_ok());
        }
        #[test]
        fn bounded_beyond_limit() {
            let scale = 0.5 * TIME_STEP.value() / stability(&$integration).real;
            assert!(matches!(
                bounded(Spectrum::Real(Time::seconds(scale)), 1.0),
                Err(IntegrationError::UnstableTimeStep(..))
            ));
        }
        #[test]
        fn bounded_safety_factor_scales_the_limit() {
            let scale = 1.5 * TIME_STEP.value() / stability(&$integration).real;
            assert!(bounded(Spectrum::Real(Time::seconds(scale)), 1.0).is_ok());
            let error = bounded(Spectrum::Real(Time::seconds(scale)), 0.5).unwrap_err();
            assert!(matches!(error, IntegrationError::UnstableTimeStep(..)));
            assert!(format!("{error}").contains("stability limit"));
        }
        #[test]
        fn bounded_imaginary_spectrum() {
            let imaginary = stability(&$integration).imaginary;
            let scale = if imaginary > 0.0 {
                0.5 * TIME_STEP.value() / imaginary
            } else {
                1.0
            };
            assert!(matches!(
                bounded(Spectrum::Imaginary(Time::seconds(scale)), 1.0),
                Err(IntegrationError::UnstableTimeStep(..))
            ));
        }
        #[test]
        fn bounded_limit_is_where_the_method_diverges() {
            let rate = stability(&$integration).real / TIME_STEP.value();
            let last = |rate: Scalar| {
                let (_, y, _): (Times, TensorVector<Quantity>, TensorVector<Quantity<Rate>>) =
                    $integration
                        .integrate(
                            |_: Quantity<Time>, x: &Quantity| Ok(x * -Rate::per_second(rate)),
                            &[Quantity::new(0.0), Quantity::new(8.0)],
                            Quantity::new(1.0),
                        )
                        .unwrap();
                y.iter().last().unwrap().value().abs()
            };
            assert!(last(0.9 * rate) <= 1.0);
            assert!(last(1.2 * rate) > 1.0);
        }
    };
}
pub(crate) use test_explicit_fixed_step;
