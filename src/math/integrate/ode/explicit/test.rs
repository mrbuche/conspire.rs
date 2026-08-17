macro_rules! test_explicit {
    ($integration: expr) => {
        use crate::math::{
            Quantity, TensorVector,
            assert::AssertionError,
            integrate::{Explicit, IntegrationError, Times},
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
    };
}
pub(crate) use test_explicit;
