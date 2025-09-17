use super::{
    super::{
        super::{
            // TensorArray, TensorRank1, TensorRank2,
            test::{TestError, assert_eq_within_tols},
        },
        // test::{rosenbrock, rosenbrock_derivative, rosenbrock_second_derivative},
    },
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
    // The whole banana region (including (-1, 1), (1, 1), and path between them) is non-convex.
    // Probably need to detect and regularize non-hyperbolic regions when using Newton's Method.
    //
    // #[test]
    // fn rosenbrock_2d() -> Result<(), TestError> {
    //     assert_eq_within_tols(
    //         &NewtonRaphson::default().minimize(
    //             rosenbrock,
    //             rosenbrock_derivative,
    //             |x: &TensorRank1<2, 1>| {
    //                 Ok(TensorRank2::<2, 1, 1>::new([
    //                     [
    //                         2.0 + 400.0 * (x[1] - x[0].powi(2)) - 800.0 * x[0].powi(2),
    //                         -400.0 * x[0],
    //                     ],
    //                     [-400.0 * x[0], 200.0],
    //                 ]))
    //             },
    //             // rosenbrock_second_derivative::<_, TensorRank2<2, 1, 1>>,
    //             TensorRank1::new([-1.0, 1.0]),
    //             EqualityConstraint::None,
    //             None,
    //         )?,
    //         &TensorRank1::<2, 1>::identity(),
    //     )
    // }
    mod line_search {
        use super::*;
        #[test]
        fn armijo() -> Result<(), TestError> {
            assert_eq_within_tols(
                &NewtonRaphson {
                    line_search: LineSearch::Armijo {
                        control: CONTROL_1,
                        cut_back: CUT_BACK,
                        max_steps: MAX_STEPS,
                    },
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
                    line_search: LineSearch::Goldstein {
                        control: CONTROL_1,
                        cut_back: CUT_BACK,
                        max_steps: MAX_STEPS,
                    },
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
                        line_search: LineSearch::Wolfe {
                            control_1: CONTROL_1,
                            control_2: CONTROL_2,
                            cut_back: CUT_BACK,
                            max_steps: MAX_STEPS,
                            strong: true,
                        },
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
                        line_search: LineSearch::Wolfe {
                            control_1: CONTROL_1,
                            control_2: CONTROL_2,
                            cut_back: CUT_BACK,
                            max_steps: MAX_STEPS,
                            strong: false,
                        },
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
