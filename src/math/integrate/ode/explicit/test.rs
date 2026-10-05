macro_rules! test_explicit {
    ($integration: expr) => {
        use crate::math::{
            Quantity, TensorVector,
            assert::AssertionError,
            integrate::{Explicit, IntegrationError, Spectrum, Times},
        };
        use crate::units::{Rate, Time};
        const RATE: Quantity<Rate> = Rate::per_second(1.0);
        #[test]
        #[should_panic(expected = "The time must contain at least two entries.")]
        fn initial_time_not_less_than_final_time() {
            let _: (Times, TensorVector<Quantity>, TensorVector<Quantity<Rate>>) = $integration
                .integrate(
                    |_: Quantity<Time>, _: &Quantity| panic!(),
                    &[Quantity::new(0.0)],
                    Quantity::new(0.0),
                )
                .unwrap();
        }
        #[test]
        fn into_test_error() {
            let result: Result<
                (Times, TensorVector<Quantity>, TensorVector<Quantity<Rate>>),
                IntegrationError,
            > = $integration.integrate(
                |_: Quantity<Time>, _: &Quantity| panic!(),
                &[Quantity::new(0.0)],
                Quantity::new(0.0),
            );
            let _: AssertionError = result.unwrap_err().into();
        }
        #[test]
        #[should_panic(expected = "The initial time must precede the final time.")]
        fn length_time_less_than_two() {
            let _: (Times, TensorVector<Quantity>, TensorVector<Quantity<Rate>>) = $integration
                .integrate(
                    |_: Quantity<Time>, _: &Quantity| panic!(),
                    &[Quantity::new(0.0), Quantity::new(1.0), Quantity::new(0.0)],
                    Quantity::new(0.0),
                )
                .unwrap();
        }
        #[test]
        fn bounded_loose_bound_agrees() -> Result<(), AssertionError> {
            let (time, solution, _): (Times, TensorVector<Quantity>, TensorVector<Quantity<Rate>>) =
                $integration.integrate(
                    |_: Quantity<Time>, x: &Quantity| Ok(x * -RATE),
                    &[Quantity::new(0.0), Quantity::new(0.8)],
                    Quantity::new(1.0),
                )?;
            let (time_bounded, solution_bounded, _): (
                Times,
                TensorVector<Quantity>,
                TensorVector<Quantity<Rate>>,
            ) = $integration.integrate_bounded(
                |_: Quantity<Time>, x: &Quantity| Ok(x * -RATE),
                |_, _| Ok(Spectrum::Real(Time::seconds(1e30))),
                1.0,
                &[Quantity::new(0.0), Quantity::new(0.8)],
                Quantity::new(1.0),
            )?;
            assert_eq!(time.len(), time_bounded.len());
            time.iter()
                .zip(time_bounded.iter())
                .try_for_each(|(a, b)| {
                    $crate::math::assert::Assert::default().eq_within_tols(a, b)
                })?;
            solution
                .iter()
                .zip(solution_bounded.iter())
                .try_for_each(|(a, b)| $crate::math::assert::Assert::default().eq_within_tols(a, b))
        }
        #[test]
        fn bounded_invalid_safety_factor() {
            [0.0, -1.0, 1.5, f64::NAN].iter().for_each(|&safety| {
                let result: Result<
                    (Times, TensorVector<Quantity>, TensorVector<Quantity<Rate>>),
                    IntegrationError,
                > = $integration.integrate_bounded(
                    |_: Quantity<Time>, x: &Quantity| Ok(x * -RATE),
                    |_, _| Ok(Spectrum::Real(Time::seconds(1.0))),
                    safety,
                    &[Quantity::new(0.0), Quantity::new(0.8)],
                    Quantity::new(1.0),
                );
                let error = result.unwrap_err();
                assert!(matches!(error, IntegrationError::InvalidSafetyFactor(_)));
                assert!(format!("{error}").contains("safety factor"));
            });
        }
    };
}
pub(crate) use test_explicit;
