use super::{
    super::{
        super::{
            // TensorArray, TensorRank1, TensorRank2,
            assert::AssertionError,
        },
        // test::{rosenbrock, rosenbrock_derivative, rosenbrock_second_derivative},
    },
    EqualityConstraint, FirstOrderRootFinding, LineSearch, NewtonRaphson, Scalar,
    SecondOrderOptimization,
};
use crate::math::assert::Assert;

const CONTROL_1: Scalar = 1e-3;
const CONTROL_2: Scalar = 1e-1;
const CUT_BACK: Scalar = 9e-1;
const MAX_STEPS: usize = 25;

mod minimize {
    use super::*;
    #[test]
    fn quadratic() -> Result<(), AssertionError> {
        Assert::default().zero_within_tols(&NewtonRaphson::default().minimize(
            |x: &Scalar| Ok(x.powi(2) / 2.0),
            |x: &Scalar| Ok(*x),
            |_: &Scalar| Ok(1.0),
            1.0,
            EqualityConstraint::None,
            None,
        )?)
    }
    //
    // "The global minimum is inside a long, narrow, parabolic-shaped flat valley.
    //  To find the valley is trivial.
    //  To converge to the global minimum, however, is difficult."
    // The whole banana region (including (-1, 1), (1, 1), and path between them) is non-convex.
    // Probably need to detect and regularize non-hyperbolic regions when using Newton's Method.
    //
    // #[test]
    // fn rosenbrock_2d() -> Result<(), AssertionError> {
    //     Assert::default().eq_within_tols(
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
        fn armijo() -> Result<(), AssertionError> {
            Assert::default().zero_within_tols(
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
            )
        }
        #[test]
        fn goldstein() -> Result<(), AssertionError> {
            Assert::default().zero_within_tols(
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
            )
        }
        mod wolfe {
            use super::*;
            #[test]
            fn strong() -> Result<(), AssertionError> {
                Assert::default().zero_within_tols(
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
                )
            }
            #[test]
            fn weak() -> Result<(), AssertionError> {
                Assert::default().zero_within_tols(
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
                )
            }
        }
    }
}

mod root {
    use super::*;
    #[test]
    fn linear() -> Result<(), AssertionError> {
        Assert::default().zero_within_tols(&NewtonRaphson::default().root(
            |x: &Scalar| Ok(*x),
            |_: &Scalar| Ok(1.0),
            1.0,
            EqualityConstraint::None,
            None,
        )?)
    }
}

mod constrained {
    use super::*;
    use crate::math::{Matrix, SquareMatrix, Vector};

    fn constraint() -> EqualityConstraint {
        let mut matrix = Matrix::zero(1, 2);
        matrix[0][0] = 1.0;
        matrix[0][1] = 1.0;
        EqualityConstraint::Linear(matrix, Vector::from([2.0]))
    }

    fn minimized(line_search: LineSearch) -> Result<Vector, AssertionError> {
        Ok(NewtonRaphson {
            line_search,
            ..Default::default()
        }
        .minimize(
            |x: &Vector| Ok((x[0].powi(2) + x[1].powi(2)) / 2.0),
            |x: &Vector| Ok(x.clone()),
            |_: &Vector| Ok(SquareMatrix::from([[1.0, 0.0], [0.0, 1.0]])),
            Vector::from([4.0, -3.0]),
            constraint(),
            None,
        )?)
    }

    #[test]
    fn none() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(&minimized(LineSearch::None)?, &Vector::from([1.0, 1.0]))
    }

    #[test]
    fn armijo() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &minimized(LineSearch::Armijo {
                control: CONTROL_1,
                cut_back: CUT_BACK,
                max_steps: MAX_STEPS,
            })?,
            &Vector::from([1.0, 1.0]),
        )
    }

    #[test]
    fn goldstein() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &minimized(LineSearch::Goldstein {
                control: CONTROL_2,
                cut_back: CUT_BACK,
                max_steps: MAX_STEPS,
            })?,
            &Vector::from([1.0, 1.0]),
        )
    }

    //
    // The Newton step of a square root overshoots ever further from far out,
    // so that the solver reaches the solution only if it shortens the step.
    //
    fn overshooting(line_search: LineSearch) -> Result<Vector, super::super::OptimizationError> {
        let mut matrix = Matrix::zero(1, 2);
        matrix[0][1] = 1.0;
        NewtonRaphson {
            line_search,
            max_steps: 100,
            ..Default::default()
        }
        .minimize(
            |x: &Vector| Ok((1.0 + x[0].powi(2)).sqrt() + x[1].powi(2) / 2.0),
            |x: &Vector| Ok(Vector::from([x[0] / (1.0 + x[0].powi(2)).sqrt(), x[1]])),
            |x: &Vector| {
                Ok(SquareMatrix::from([
                    [(1.0 + x[0].powi(2)).powf(-1.5), 0.0],
                    [0.0, 1.0],
                ]))
            },
            Vector::from([2.0, 0.0]),
            EqualityConstraint::Linear(matrix, Vector::zero(1)),
            None,
        )
    }

    #[test]
    fn overshooting_armijo() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &overshooting(LineSearch::Armijo {
                control: CONTROL_1,
                cut_back: CUT_BACK,
                max_steps: MAX_STEPS,
            })?,
            &Vector::zero(2),
        )
    }

    #[test]
    fn overshooting_none() {
        //
        // Left unshortened the step runs away, until the square root overflows
        // and the solver mistakes the zero residual that follows for success.
        //
        assert!(match overshooting(LineSearch::None) {
            Ok(solution) => solution[0].abs() > 1.0,
            Err(_) => true,
        })
    }
}
