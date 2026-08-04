use super::{
    super::{
        super::{
            // TensorArray, TensorRank1, TensorRank2,
            assert::AssertionError,
        },
        // test::{rosenbrock, rosenbrock_derivative, rosenbrock_second_derivative},
    },
    EqualityConstraint, FirstOrderRootFinding, LineSearch, NewtonRaphson, OptimizationError,
    Scalar, SecondOrderOptimization, TrustRegion,
};
use crate::math::{Norm, Tensor, assert::Assert};

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

    #[test]
    fn error() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &minimized(LineSearch::Error {
                cut_back: CUT_BACK,
                max_steps: MAX_STEPS,
            })?,
            &Vector::from([1.0, 1.0]),
        )
    }

    /// Root finding has no merit function, so every other line search panics
    /// on the residual it would need. Backtracking for errors asks for none.
    #[test]
    fn error_root() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &NewtonRaphson {
                line_search: LineSearch::Error {
                    cut_back: CUT_BACK,
                    max_steps: MAX_STEPS,
                },
                ..Default::default()
            }
            .root(
                |x: &Vector| Ok(x.clone()),
                |_: &Vector| Ok(SquareMatrix::from([[1.0, 0.0], [0.0, 1.0]])),
                Vector::from([4.0, -3.0]),
                constraint(),
                None,
            )?,
            &Vector::from([1.0, 1.0]),
        )
    }

    /// The step is shortened until the residual can be evaluated where it
    /// lands, so a region it cannot be evaluated in is stepped around.
    ///
    /// The tangent understates the curvature fourfold, so the full step from
    /// the initial guess overshoots the root and lands beyond the barrier.
    /// Without backtracking the residual is never evaluable again.
    fn barrier(line_search: LineSearch) -> Result<Vector, super::super::OptimizationError> {
        NewtonRaphson {
            line_search,
            ..Default::default()
        }
        .root(
            |x: &Vector| {
                if x[0] < 3.0 {
                    Err("Beyond the barrier.".to_string())
                } else {
                    Ok(Vector::from([x[0] - 4.0, x[1]]))
                }
            },
            |_: &Vector| Ok(SquareMatrix::from([[0.25, 0.0], [0.0, 1.0]])),
            Vector::from([6.0, -3.0]),
            {
                let mut matrix = Matrix::zero(1, 2);
                matrix[0][1] = 1.0;
                EqualityConstraint::Linear(matrix, Vector::from([1.0]))
            },
            None,
        )
    }

    #[test]
    fn error_backtracks() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &barrier(LineSearch::Error {
                cut_back: 5e-1,
                max_steps: MAX_STEPS,
            })?,
            &Vector::from([4.0, 1.0]),
        )
    }

    #[test]
    fn error_backtracks_needed() {
        assert!(barrier(LineSearch::None).is_err())
    }

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
        assert!(match overshooting(LineSearch::None) {
            Ok(solution) => solution[0].abs() > 1.0,
            Err(_) => true,
        })
    }

    /// The same overshooting, met with a limit on the step instead of a line
    /// search, and by root finding rather than minimization.
    ///
    /// Newton on this residual multiplies the distance from the root by the
    /// square of it, so any start beyond one diverges. Nothing about that step
    /// fails to evaluate, so backtracking for errors would let it through.
    fn steep(
        trust_region: TrustRegion,
        line_search: LineSearch,
    ) -> Result<Vector, OptimizationError> {
        let mut matrix = Matrix::zero(1, 2);
        matrix[0][1] = 1.0;
        NewtonRaphson {
            line_search,
            trust_region,
            max_steps: 100,
            ..Default::default()
        }
        .root(
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
    fn trust_region() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &steep(
                TrustRegion::Fixed {
                    radius: 0.75,
                    norm: Norm::Chebyshev,
                },
                LineSearch::None,
            )?,
            &Vector::zero(2),
        )
    }

    #[test]
    fn trust_region_needed() {
        assert!(match steep(TrustRegion::None, LineSearch::None) {
            Ok(solution) => solution[0].abs() > 1.0,
            Err(_) => true,
        })
    }

    /// The limit is measured in whichever norm is asked for, and the two
    /// disagree by the square root of the number of variables.
    ///
    /// Every variable starts the same distance from its root, so Chebyshev
    /// sees a step of one where Euclidean sees one of ten. The same limit is
    /// therefore ten times tighter in the second, which is the whole reason
    /// the step has a norm of its own rather than the one errors are measured
    /// in.
    fn wide(norm: Norm) -> Result<Vector, OptimizationError> {
        const WIDTH: usize = 100;
        let mut constraint_matrix = Matrix::zero(1, WIDTH);
        constraint_matrix[0][WIDTH - 1] = 1.0;
        let mut initial_guess = Vector::zero(WIDTH);
        initial_guess
            .iter_mut()
            .take(WIDTH - 1)
            .for_each(|entry| *entry = 1.0);
        let mut tangent = SquareMatrix::zero(WIDTH);
        (0..WIDTH).for_each(|i| tangent[i][i] = 1.0);
        NewtonRaphson {
            max_steps: 10,
            trust_region: TrustRegion::Fixed { radius: 5e-1, norm },
            ..Default::default()
        }
        .root(
            |x: &Vector| Ok(x.clone()),
            |_: &Vector| Ok(tangent.clone()),
            initial_guess,
            EqualityConstraint::Linear(constraint_matrix, Vector::zero(1)),
            None,
        )
    }

    #[test]
    fn trust_region_norm_chebyshev() -> Result<(), AssertionError> {
        Assert::default().zero_within_tols(&wide(Norm::Chebyshev)?)
    }

    #[test]
    fn trust_region_norm_euclidean() {
        assert!(wide(Norm::Euclidean).is_err())
    }

    /// The step that diverges here is a perfectly good one to evaluate, so
    /// only the limit catches it.
    #[test]
    fn trust_region_beyond_errors() {
        assert!(match steep(
            TrustRegion::None,
            LineSearch::Error {
                cut_back: 5e-1,
                max_steps: MAX_STEPS,
            },
        ) {
            Ok(solution) => solution[0].abs() > 1.0,
            Err(_) => true,
        })
    }
}
