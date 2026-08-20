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
    //             |x: &TensorRank1<2, Current>| {
    //                 Ok(TensorRank2::<2, Current, Current>::new([
    //                     [
    //                         2.0 + 400.0 * (x[1] - x[0].powi(2)) - 800.0 * x[0].powi(2),
    //                         -400.0 * x[0],
    //                     ],
    //                     [-400.0 * x[0], 200.0],
    //                 ]))
    //             },
    //             // rosenbrock_second_derivative::<_, TensorRank2<2, Current, Current>>,
    //             TensorRank1::new([-1.0, 1.0]),
    //             EqualityConstraint::None,
    //             None,
    //         )?,
    //         &TensorRank1::<2, Current>::identity(),
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
    use crate::math::{SquareMatrix, Vector, sparse::SparseSolver};
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
    fn coupled(sparse: Option<SparseSolver>) -> Result<Vector, AssertionError> {
        Ok(NewtonRaphson::default().root(
            |x: &Vector| {
                Ok(Vector::from([
                    x[0] + 2.0 * x[1] - 5.0,
                    3.0 * x[0] - x[1] - 1.0,
                ]))
            },
            |_: &Vector| Ok(SquareMatrix::from([[1.0, 2.0], [3.0, -1.0]])),
            Vector::from([0.0, 0.0]),
            EqualityConstraint::None,
            sparse,
        )?)
    }
    #[test]
    fn coupled_dense() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(&coupled(None)?, &Vector::from([1.0, 2.0]))
    }
    #[test]
    fn coupled_sparse() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &coupled(Some(SparseSolver::from_pattern(
                2,
                vec![(0, 0), (0, 1), (1, 0), (1, 1)],
                false,
            )))?,
            &Vector::from([1.0, 2.0]),
        )
    }
}

mod constrained {
    use super::*;
    use crate::math::{Matrix, SquareMatrix, Vector, optimize::Tolerances};

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

    fn minimized_trust_region(trust_region: TrustRegion) -> Result<Vector, AssertionError> {
        Ok(NewtonRaphson {
            trust_region,
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
    fn trust_region_adaptive() -> Result<(), AssertionError> {
        // radius starts far smaller than the distance to the minimum, so
        // convergence within max_steps requires the accept/reject loop to
        // actually grow the radius across several outer Newton iterations;
        // kept above the constraint's own feasibility floor (~0.71 here,
        // from the initial constraint violation) so more_sorensen isn't
        // being asked for a radius no shift could ever reach
        Assert::default().eq_within_tols(
            &minimized_trust_region(TrustRegion::Adaptive {
                radius: 1.0,
                max_radius: 10.0,
            })?,
            &Vector::from([1.0, 1.0]),
        )
    }

    fn scaled(rel_tol: Option<Scalar>) -> Result<Vector, OptimizationError> {
        const SCALE: Scalar = 1e12;
        NewtonRaphson {
            abs_tol: Tolerances {
                constraint: 0.0,
                residual: 0.0,
            },
            rel_tol,
            ..Default::default()
        }
        .minimize(
            |x: &Vector| Ok(SCALE * (x[0].powi(2) + x[1].powi(2)) / 2.0),
            |x: &Vector| Ok(x * SCALE),
            |_: &Vector| Ok(SquareMatrix::from([[SCALE, 0.0], [0.0, SCALE]])),
            Vector::from([4.0, -3.0]),
            constraint(),
            None,
        )
    }

    #[test]
    fn relative() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(&scaled(Some(1e-8))?, &Vector::from([1.0, 1.0]))
    }

    #[test]
    fn relative_is_what_absolute_cannot_be() {
        assert!(scaled(None).is_err())
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

mod fixed {
    use super::*;
    use crate::math::{SquareMatrix, Vector};
    #[test]
    fn dense() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &NewtonRaphson::default().root(
                |x: &Vector| Ok(Vector::from([x[0] - 4.0, x[1] - 7.0])),
                |_: &Vector| Ok(SquareMatrix::from([[1.0, 0.0], [0.0, 1.0]])),
                Vector::from([0.0, 3.0]),
                EqualityConstraint::Fixed(vec![1]),
                None,
            )?,
            &Vector::from([4.0, 3.0]),
        )
    }
}
