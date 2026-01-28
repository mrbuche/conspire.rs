macro_rules! test_explicit {
    ($integration: expr) => {
        use crate::math::{
            Scalar, Vector,
            integrate::{Explicit, IntegrationError},
            test::TestError,
        };
        #[test]
        #[should_panic(expected = "The time must contain at least two entries.")]
        fn initial_time_not_less_than_final_time() {
            let _: (Vector, Vector, _) = $integration
                .integrate(|_: Scalar, _: &Scalar| panic!(), &[0.0], 0.0)
                .unwrap();
        }
        #[test]
        fn into_test_error() {
            let result: Result<(Vector, Vector, _), IntegrationError> =
                $integration.integrate(|_: Scalar, _: &Scalar| panic!(), &[0.0], 0.0);
            let _: TestError = result.unwrap_err().into();
        }
        #[test]
        #[should_panic(expected = "The initial time must precede the final time.")]
        fn length_time_less_than_two() {
            let _: (Vector, Vector, _) = $integration
                .integrate(|_: Scalar, _: &Scalar| panic!(), &[0.0, 1.0, 0.0], 0.0)
                .unwrap();
        }
    };
}
pub(crate) use test_explicit;
