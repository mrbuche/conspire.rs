use super::{
    super::super::test::{TestError, assert_eq_within_tols},
    EqualityConstraint, FirstOrderRootFinding, LineSearch, NewtonRaphson, Scalar,
    SecondOrderOptimization,
};

const CONTROL_1: Scalar = 1e-3;
const CONTROL_2: Scalar = 1e-1;
const CUT_BACK: Scalar = 9e-1;
const MAX_STEPS: usize = 25;

mod minimize {
    use super::*;
    #[test]
    fn quadratic() -> Result<(), TestError> {
        assert_eq_within_tols(
            &NewtonRaphson::default().minimize(
                |x: &Scalar| Ok(x.powi(2) / 2.0),
                |x: &Scalar| Ok(*x),
                |_: &Scalar| Ok(1.0),
                1.0,
                EqualityConstraint::None,
                None,
            )?,
            &0.0,
        )
    }
    //
    // "The global minimum is inside a long, narrow, parabolic-shaped flat valley.
    //  To find the valley is trivial.
    //  To converge to the global minimum, however, is difficult."
    // Probably need to detect and regularize non-hyperbolic regions when using Newton's Method.
    //
    // #[test]
    // fn rosenbrock() {
    //     todo!()
    // }
    mod line_search {
        use super::*;
        #[test]
        fn armijo() -> Result<(), TestError> {
            assert_eq_within_tols(
                &NewtonRaphson {
                    line_search: Some(LineSearch::Armijo(CONTROL_1, CUT_BACK, MAX_STEPS)),
                    ..Default::default()
                }
                .minimize(
                    |x: &Scalar| Ok(x.powi(2) / 2.0),
                    |x: &Scalar| Ok(*x),
                    |_: &Scalar| Ok(1.0),
                    1.0,
                    EqualityConstraint::None,
                    None,
                )?,
                &0.0,
            )
        }
        #[test]
        fn goldstein() -> Result<(), TestError> {
            assert_eq_within_tols(
                &NewtonRaphson {
                    line_search: Some(LineSearch::Goldstein(CONTROL_1, CUT_BACK, MAX_STEPS)),
                    ..Default::default()
                }
                .minimize(
                    |x: &Scalar| Ok(x.powi(2) / 2.0),
                    |x: &Scalar| Ok(*x),
                    |_: &Scalar| Ok(1.0),
                    1.0,
                    EqualityConstraint::None,
                    None,
                )?,
                &0.0,
            )
        }
        mod wolfe {
            use super::*;
            #[test]
            fn strong() -> Result<(), TestError> {
                assert_eq_within_tols(
                    &NewtonRaphson {
                        line_search: Some(LineSearch::Wolfe(
                            CONTROL_1, CONTROL_2, CUT_BACK, MAX_STEPS, true,
                        )),
                        ..Default::default()
                    }
                    .minimize(
                        |x: &Scalar| Ok(x.powi(2) / 2.0),
                        |x: &Scalar| Ok(*x),
                        |_: &Scalar| Ok(1.0),
                        1.0,
                        EqualityConstraint::None,
                        None,
                    )?,
                    &0.0,
                )
            }
            #[test]
            fn weak() -> Result<(), TestError> {
                assert_eq_within_tols(
                    &NewtonRaphson {
                        line_search: Some(LineSearch::Wolfe(
                            CONTROL_1, CONTROL_2, CUT_BACK, MAX_STEPS, false,
                        )),
                        ..Default::default()
                    }
                    .minimize(
                        |x: &Scalar| Ok(x.powi(2) / 2.0),
                        |x: &Scalar| Ok(*x),
                        |_: &Scalar| Ok(1.0),
                        1.0,
                        EqualityConstraint::None,
                        None,
                    )?,
                    &0.0,
                )
            }
        }
    }
}

mod root {
    use super::*;
    #[test]
    fn linear() -> Result<(), TestError> {
        assert_eq_within_tols(
            &NewtonRaphson::default().root(
                |x: &Scalar| Ok(*x),
                |_: &Scalar| Ok(1.0),
                1.0,
                EqualityConstraint::None,
            )?,
            &0.0,
        )
    }
}
