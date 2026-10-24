use super::{
    super::{
        super::{
            // TensorArray, TensorRank1, TensorRank2,
            assert::AssertionError,
        },
        // test::{rosenbrock, rosenbrock_derivative, rosenbrock_second_derivative},
    },
    Direct, EqualityConstraint, EqualityConstraints, NewtonRaphson, NoSearch, ObjectiveMerit,
    Optimization, OptimizationError, ResidualMerit, RootFinding, Scalar, TrustRegion,
};
use crate::math::{
    Norm, Tensor,
    assert::Assert,
    optimize::{Armijo, Error, Goldstein, Residual, Wolfe},
};

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
            vec![EqualityConstraint::None],
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
                &NewtonRaphson::default()
                    .with_line_search(Armijo {
                        control: CONTROL_1,
                        cut_back: CUT_BACK,
                        max_steps: MAX_STEPS,
                    })
                    .minimize(
                        |x: &Scalar| Ok(x.powi(2) / 2.0),
                        |x: &Scalar| Ok(*x),
                        |_: &Scalar| Ok(1.0),
                        1.0,
                        vec![EqualityConstraint::None],
                        None,
                    )?,
            )
        }
        #[test]
        fn goldstein() -> Result<(), AssertionError> {
            Assert::default().zero_within_tols(
                &NewtonRaphson::default()
                    .with_line_search(Goldstein {
                        control: CONTROL_1,
                        cut_back: CUT_BACK,
                        max_steps: MAX_STEPS,
                    })
                    .minimize(
                        |x: &Scalar| Ok(x.powi(2) / 2.0),
                        |x: &Scalar| Ok(*x),
                        |_: &Scalar| Ok(1.0),
                        1.0,
                        vec![EqualityConstraint::None],
                        None,
                    )?,
            )
        }
        mod wolfe {
            use super::*;
            #[test]
            fn strong() -> Result<(), AssertionError> {
                Assert::default().zero_within_tols(
                    &NewtonRaphson::default()
                        .with_line_search(Wolfe {
                            control_1: CONTROL_1,
                            control_2: CONTROL_2,
                            cut_back: CUT_BACK,
                            max_steps: MAX_STEPS,
                            strong: true,
                        })
                        .minimize(
                            |x: &Scalar| Ok(x.powi(2) / 2.0),
                            |x: &Scalar| Ok(*x),
                            |_: &Scalar| Ok(1.0),
                            1.0,
                            vec![EqualityConstraint::None],
                            None,
                        )?,
                )
            }
            #[test]
            fn weak() -> Result<(), AssertionError> {
                Assert::default().zero_within_tols(
                    &NewtonRaphson::default()
                        .with_line_search(Wolfe {
                            control_1: CONTROL_1,
                            control_2: CONTROL_2,
                            cut_back: CUT_BACK,
                            max_steps: MAX_STEPS,
                            strong: false,
                        })
                        .minimize(
                            |x: &Scalar| Ok(x.powi(2) / 2.0),
                            |x: &Scalar| Ok(*x),
                            |_: &Scalar| Ok(1.0),
                            1.0,
                            vec![EqualityConstraint::None],
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
            vec![EqualityConstraint::None],
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
            vec![EqualityConstraint::None],
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
    fn minimized(line_search: impl ObjectiveMerit) -> Result<Vector, AssertionError> {
        Ok(NewtonRaphson::default()
            .with_line_search(line_search)
            .minimize(
                |x: &Vector| Ok((x[0].powi(2) + x[1].powi(2)) / 2.0),
                |x: &Vector| Ok(x.clone()),
                |_: &Vector| Ok(SquareMatrix::from([[1.0, 0.0], [0.0, 1.0]])),
                Vector::from([4.0, -3.0]),
                vec![constraint()],
                None,
            )?)
    }
    #[test]
    fn none() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(&minimized(NoSearch)?, &Vector::from([1.0, 1.0]))
    }
    fn minimized_trust_region(trust_region: TrustRegion) -> Result<Vector, AssertionError> {
        Ok(NewtonRaphson {
            trust_region,
            linear_solver: Direct,
            ..Default::default()
        }
        .minimize(
            |x: &Vector| Ok((x[0].powi(2) + x[1].powi(2)) / 2.0),
            |x: &Vector| Ok(x.clone()),
            |_: &Vector| Ok(SquareMatrix::from([[1.0, 0.0], [0.0, 1.0]])),
            Vector::from([4.0, -3.0]),
            vec![constraint()],
            None,
        )?)
    }
    #[test]
    fn trust_region_adaptive() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &minimized_trust_region(TrustRegion::Adaptive {
                radius: 1.0,
                max_radius: 10.0,
            })?,
            &Vector::from([1.0, 1.0]),
        )
    }
    #[test]
    fn trust_region_adaptive_failed_trial_is_a_rejection() -> Result<(), AssertionError> {
        let root = |x: Scalar| (1.0 + x * x).sqrt();
        let result = NewtonRaphson {
            trust_region: TrustRegion::Adaptive {
                radius: 100.0,
                max_radius: 100.0,
            },
            linear_solver: Direct,
            max_steps: 100,
            ..Default::default()
        }
        .minimize(
            |x: &Vector| {
                if x.iter().any(|x_i| x_i.abs() > 10.0) {
                    Err("out of domain".to_string())
                } else {
                    Ok(root(x[0]) + root(x[1]))
                }
            },
            |x: &Vector| Ok(x.iter().map(|&x_i| x_i / root(x_i)).collect()),
            |x: &Vector| {
                let mut hessian = SquareMatrix::zero(2);
                (0..2).for_each(|i| hessian[i][i] = root(x[i]).powi(-3));
                Ok(hessian)
            },
            Vector::from([4.0, -3.0]),
            vec![constraint()],
            None,
        )?;
        Assert::default().eq_within_tols(&result, &Vector::from([1.0, 1.0]))
    }
    fn scaled(rel_tol: Option<Scalar>) -> Result<Vector, OptimizationError> {
        const SCALE: Scalar = 1e12;
        NewtonRaphson {
            abs_tol: Tolerances {
                constraint: 0.0,
                residual: 0.0,
            },
            rel_tol,
            linear_solver: Direct,
            ..Default::default()
        }
        .minimize(
            |x: &Vector| Ok(SCALE * (x[0].powi(2) + x[1].powi(2)) / 2.0),
            |x: &Vector| Ok(x * SCALE),
            |_: &Vector| Ok(SquareMatrix::from([[SCALE, 0.0], [0.0, SCALE]])),
            Vector::from([4.0, -3.0]),
            vec![constraint()],
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
            &minimized(Armijo {
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
            &minimized(Goldstein {
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
            &minimized(Error {
                cut_back: CUT_BACK,
                max_steps: MAX_STEPS,
            })?,
            &Vector::from([1.0, 1.0]),
        )
    }
    #[test]
    fn error_root() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &NewtonRaphson::default()
                .with_line_search(Error {
                    cut_back: CUT_BACK,
                    max_steps: MAX_STEPS,
                })
                .root(
                    |x: &Vector| Ok(x.clone()),
                    |_: &Vector| Ok(SquareMatrix::from([[1.0, 0.0], [0.0, 1.0]])),
                    Vector::from([4.0, -3.0]),
                    vec![constraint()],
                    None,
                )?,
            &Vector::from([1.0, 1.0]),
        )
    }
    fn barrier(line_search: impl ResidualMerit) -> Result<Vector, super::super::OptimizationError> {
        NewtonRaphson::default().with_line_search(line_search).root(
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
                vec![EqualityConstraint::Linear(matrix, Vector::from([1.0]))]
            },
            None,
        )
    }
    #[test]
    fn error_backtracks() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &barrier(Error {
                cut_back: 5e-1,
                max_steps: MAX_STEPS,
            })?,
            &Vector::from([4.0, 1.0]),
        )
    }
    #[test]
    fn error_backtracks_needed() {
        assert!(barrier(NoSearch).is_err())
    }
    fn overshooting(
        line_search: impl ObjectiveMerit,
    ) -> Result<Vector, super::super::OptimizationError> {
        let mut matrix = Matrix::zero(1, 2);
        matrix[0][1] = 1.0;
        NewtonRaphson {
            max_steps: 100,
            linear_solver: Direct,
            ..Default::default()
        }
        .with_line_search(line_search)
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
            vec![EqualityConstraint::Linear(matrix, Vector::zero(1))],
            None,
        )
    }
    #[test]
    fn overshooting_armijo() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &overshooting(Armijo {
                control: CONTROL_1,
                cut_back: CUT_BACK,
                max_steps: MAX_STEPS,
            })?,
            &Vector::zero(2),
        )
    }
    #[test]
    fn overshooting_none() {
        assert!(match overshooting(NoSearch) {
            Ok(solution) => solution[0].abs() > 1.0,
            Err(_) => true,
        })
    }
    fn steep(
        trust_region: TrustRegion,
        line_search: impl ResidualMerit,
    ) -> Result<Vector, OptimizationError> {
        let mut matrix = Matrix::zero(1, 2);
        matrix[0][1] = 1.0;
        NewtonRaphson {
            trust_region,
            max_steps: 100,
            linear_solver: Direct,
            ..Default::default()
        }
        .with_line_search(line_search)
        .root(
            |x: &Vector| Ok(Vector::from([x[0] / (1.0 + x[0].powi(2)).sqrt(), x[1]])),
            |x: &Vector| {
                Ok(SquareMatrix::from([
                    [(1.0 + x[0].powi(2)).powf(-1.5), 0.0],
                    [0.0, 1.0],
                ]))
            },
            Vector::from([2.0, 0.0]),
            vec![EqualityConstraint::Linear(matrix, Vector::zero(1))],
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
                NoSearch,
            )?,
            &Vector::zero(2),
        )
    }
    #[test]
    fn trust_region_needed() {
        assert!(match steep(TrustRegion::None, NoSearch) {
            Ok(solution) => solution[0].abs() > 1.0,
            Err(_) => true,
        })
    }
    #[test]
    fn residual_needs_no_trust_region() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &steep(
                TrustRegion::None,
                Residual {
                    control: CONTROL_1,
                    cut_back: 5e-1,
                    max_steps: MAX_STEPS,
                },
            )?,
            &Vector::zero(2),
        )
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
            linear_solver: Direct,
            ..Default::default()
        }
        .root(
            |x: &Vector| Ok(x.clone()),
            |_: &Vector| Ok(tangent.clone()),
            initial_guess,
            vec![EqualityConstraint::Linear(
                constraint_matrix,
                Vector::zero(1),
            )],
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
            Error {
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
                vec![EqualityConstraint::Fixed(vec![1])],
                None,
            )?,
            &Vector::from([4.0, 3.0]),
        )
    }
}

mod prescribed {
    use super::*;
    use crate::math::{
        Matrix, SquareMatrix, Vector,
        optimize::{LinearSolver, Prescription},
    };
    fn residual(x: &Vector) -> Result<Vector, String> {
        Ok([
            2.0 * x[0] - x[1] - 1.0,
            -x[0] + 2.0 * x[1] - x[2],
            -x[1] + 2.0 * x[2] - 1.0,
        ]
        .into())
    }
    fn tangent(_: &Vector) -> Result<SquareMatrix, String> {
        Ok([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]].into())
    }
    fn energy(x: &Vector) -> Result<Scalar, String> {
        Ok(0.5
            * (2.0 * x[0] * x[0] - 2.0 * x[0] * x[1] + 2.0 * x[1] * x[1] - 2.0 * x[1] * x[2]
                + 2.0 * x[2] * x[2])
            - x[0]
            - x[2])
    }
    fn multiplier() -> EqualityConstraint {
        let mut matrix = Matrix::zero(1, 3);
        matrix[0][2] = 1.0;
        EqualityConstraint::Linear(matrix, Vector::from([0.7]))
    }
    const SOLUTION: [Scalar; 3] = [0.9, 0.8, 0.7];
    #[test]
    fn an_absolute_value() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &NewtonRaphson::default().root(
                |x: &Vector| Ok(Vector::from([x[0] - 4.0, x[1] - 7.0])),
                |_: &Vector| Ok(SquareMatrix::from([[1.0, 0.0], [0.0, 1.0]])),
                Vector::from([0.0, 3.0]),
                vec![EqualityConstraint::Prescribed(vec![(
                    1,
                    Prescription::Absolute(5.0),
                )])],
                None,
            )?,
            &Vector::from([4.0, 5.0]),
        )
    }
    #[test]
    fn an_offset() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &NewtonRaphson::default().root(
                |x: &Vector| Ok(Vector::from([x[0] - 4.0, x[1] - 7.0])),
                |_: &Vector| Ok(SquareMatrix::from([[1.0, 0.0], [0.0, 1.0]])),
                Vector::from([0.0, 3.0]),
                vec![EqualityConstraint::Prescribed(vec![(
                    1,
                    Prescription::Offset(0.5),
                )])],
                None,
            )?,
            &Vector::from([4.0, 3.5]),
        )
    }
    #[test]
    fn a_hold_is_a_fixed_index() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &NewtonRaphson::default().root(
                |x: &Vector| Ok(Vector::from([x[0] - 4.0, x[1] - 7.0])),
                |_: &Vector| Ok(SquareMatrix::from([[1.0, 0.0], [0.0, 1.0]])),
                Vector::from([0.0, 3.0]),
                vec![EqualityConstraint::Prescribed(vec![(
                    1,
                    Prescription::Initial,
                )])],
                None,
            )?,
            &Vector::from([4.0, 3.0]),
        )
    }
    #[test]
    fn elimination_and_multipliers_find_the_same_root() -> Result<(), AssertionError> {
        let eliminated = NewtonRaphson::default().root(
            residual,
            tangent,
            Vector::zero(3),
            vec![EqualityConstraint::Prescribed(vec![(
                2,
                Prescription::Absolute(0.7),
            )])],
            None,
        )?;
        let multiplied = NewtonRaphson::default().root(
            residual,
            tangent,
            Vector::zero(3),
            vec![multiplier()],
            None,
        )?;
        Assert::default().eq_within_tols(&eliminated, &Vector::from(SOLUTION))?;
        Assert::default().eq_within_tols(&multiplied, &eliminated)
    }
    struct Diagonal;
    impl LinearSolver for Diagonal {
        type Tangent = Vector;
        fn solve(
            &self,
            tangent: Vector,
            retained: &[usize],
            residual: &Vector,
        ) -> Result<Vector, String> {
            Ok(retained
                .iter()
                .zip(residual.iter())
                .map(|(&dof, residual)| residual / tangent[dof])
                .collect())
        }
    }
    #[test]
    fn a_linear_solver_that_needs_fixed_indices_is_given_them() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &NewtonRaphson::with(Diagonal).root(
                |x: &Vector| {
                    Ok(Vector::from([
                        2.0 * (x[0] - 1.0),
                        4.0 * (x[1] - 2.0),
                        5.0 * (x[2] - 3.0),
                    ]))
                },
                |_: &Vector| Ok(Vector::from([2.0, 4.0, 5.0])),
                Vector::zero(3),
                vec![EqualityConstraint::Prescribed(vec![
                    (1, Prescription::Absolute(9.0)),
                    (2, Prescription::Offset(1.0)),
                ])],
                None,
            )?,
            &Vector::from([1.0, 9.0, 1.0]),
        )
    }
    #[test]
    fn a_linear_solver_that_needs_fixed_indices_refuses_multipliers() {
        let mut matrix = Matrix::zero(1, 3);
        matrix[0][2] = 1.0;
        assert!(
            NewtonRaphson::with(Diagonal)
                .root(
                    |x: &Vector| Ok(Vector::from([x[0], x[1], x[2]])),
                    |_: &Vector| Ok(Vector::from([1.0, 1.0, 1.0])),
                    Vector::zero(3),
                    vec![EqualityConstraint::Linear(matrix, Vector::from([0.7]))],
                    None,
                )
                .is_err()
        )
    }
    #[test]
    fn an_offset_from_the_initial_guess_finds_the_same_root() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &NewtonRaphson::default().root(
                residual,
                tangent,
                Vector::from([0.0, 0.0, 0.2]),
                vec![EqualityConstraint::Prescribed(vec![(
                    2,
                    Prescription::Offset(0.5),
                )])],
                None,
            )?,
            &Vector::from(SOLUTION),
        )
    }
    #[test]
    fn minimizing_finds_the_same_minimum_as_multipliers() -> Result<(), AssertionError> {
        let eliminated = NewtonRaphson::default().minimize(
            energy,
            residual,
            tangent,
            Vector::zero(3),
            vec![EqualityConstraint::Prescribed(vec![(
                2,
                Prescription::Absolute(0.7),
            )])],
            None,
        )?;
        let multiplied = NewtonRaphson::default().minimize(
            energy,
            residual,
            tangent,
            Vector::zero(3),
            vec![multiplier()],
            None,
        )?;
        Assert::default().eq_within_tols(&eliminated, &Vector::from(SOLUTION))?;
        Assert::default().eq_within_tols(&multiplied, &eliminated)
    }
}

mod multiplied {
    use super::*;
    use crate::math::{Matrix, SquareMatrix, Vector, optimize::Prescription, sparse::SparseSolver};
    fn residual(x: &Vector) -> Result<Vector, String> {
        Ok([
            2.0 * x[0] - x[1] - 1.0,
            -x[0] + 2.0 * x[1] - x[2],
            -x[1] + 2.0 * x[2] - 1.0,
        ]
        .into())
    }
    fn tangent(_: &Vector) -> Result<SquareMatrix, String> {
        Ok([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]].into())
    }
    fn energy(x: &Vector) -> Result<Scalar, String> {
        Ok(0.5
            * (2.0 * x[0] * x[0] - 2.0 * x[0] * x[1] + 2.0 * x[1] * x[1] - 2.0 * x[1] * x[2]
                + 2.0 * x[2] * x[2])
            - x[0]
            - x[2])
    }
    fn solver(rows: usize) -> Option<SparseSolver> {
        let mut pattern = vec![(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)];
        (0..rows).for_each(|row| {
            pattern.push((3 + row, 2));
            pattern.push((2, 3 + row));
        });
        Some(SparseSolver::from_pattern(3 + rows, pattern, true))
    }
    fn mixed() -> EqualityConstraints {
        vec![
            EqualityConstraint::Prescribed(vec![(0, Prescription::Absolute(0.5))]),
            EqualityConstraint::Multipliers(vec![(2, Prescription::Absolute(0.9))]),
        ]
    }
    const SOLUTION: [Scalar; 3] = [0.5, 0.7, 0.9];
    fn rooted(constraint: EqualityConstraints, sparse: Option<SparseSolver>) -> Vector {
        NewtonRaphson::default()
            .root(residual, tangent, Vector::zero(3), constraint, sparse)
            .unwrap()
    }
    fn minimized(constraint: EqualityConstraints, sparse: Option<SparseSolver>) -> Vector {
        NewtonRaphson::default()
            .minimize(
                energy,
                residual,
                tangent,
                Vector::zero(3),
                constraint,
                sparse,
            )
            .unwrap()
    }
    #[test]
    fn what_is_multiplied_is_held_at_its_value() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            rooted(
                vec![EqualityConstraint::Multipliers(vec![(
                    2,
                    Prescription::Absolute(0.7),
                )])],
                None,
            ),
            &Vector::from([0.9, 0.8, 0.7]),
        )
    }
    #[test]
    fn an_offset_and_a_hold_are_relative_to_the_initial_guess() -> Result<(), AssertionError> {
        let solve = |prescription| {
            NewtonRaphson::default()
                .root(
                    |x: &Vector| Ok(Vector::from([x[0] - 4.0, x[1] - 7.0])),
                    |_: &Vector| Ok(SquareMatrix::from([[1.0, 0.0], [0.0, 1.0]])),
                    Vector::from([0.0, 3.0]),
                    vec![EqualityConstraint::Multipliers(vec![(1, prescription)])],
                    None,
                )
                .unwrap()
        };
        Assert::default()
            .eq_within_tols(solve(Prescription::Offset(0.5)), &Vector::from([4.0, 3.5]))?;
        Assert::default().eq_within_tols(solve(Prescription::Initial), &Vector::from([4.0, 3.0]))
    }
    #[test]
    fn one_index_eliminated_and_another_multiplied_find_the_root() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(rooted(mixed(), None), &Vector::from(SOLUTION))?;
        Assert::default().eq_within_tols(rooted(mixed(), solver(1)), &Vector::from(SOLUTION))
    }
    #[test]
    fn one_index_eliminated_and_another_multiplied_find_the_minimum() -> Result<(), AssertionError>
    {
        Assert::default().eq_within_tols(minimized(mixed(), None), &Vector::from(SOLUTION))?;
        Assert::default().eq_within_tols(minimized(mixed(), solver(1)), &Vector::from(SOLUTION))
    }
    #[test]
    fn the_same_constraint_held_either_way_gives_the_same_answer() -> Result<(), AssertionError> {
        let entries = vec![
            (0, Prescription::Absolute(0.5)),
            (2, Prescription::Absolute(0.9)),
        ];
        let eliminated = rooted(vec![EqualityConstraint::Prescribed(entries.clone())], None);
        let multiplied = rooted(vec![EqualityConstraint::Multipliers(entries)], None);
        Assert::default().eq_within_tols(&eliminated, &Vector::from(SOLUTION))?;
        Assert::default().eq_within_tols(&multiplied, &eliminated)
    }
    #[test]
    fn an_eliminated_index_still_counts_in_a_linear_constraint() -> Result<(), AssertionError> {
        let mut matrix = Matrix::zero(1, 3);
        matrix[0][0] = 1.0;
        matrix[0][2] = 1.0;
        let constraint = vec![
            EqualityConstraint::Prescribed(vec![(0, Prescription::Absolute(0.5))]),
            EqualityConstraint::Linear(matrix, Vector::from([1.4])),
        ];
        Assert::default()
            .eq_within_tols(rooted(constraint.clone(), None), &Vector::from(SOLUTION))?;
        Assert::default().eq_within_tols(rooted(constraint, solver(1)), &Vector::from(SOLUTION))
    }
    #[test]
    fn the_last_entry_wins_whichever_way_it_is_held() -> Result<(), AssertionError> {
        let constraint = vec![
            EqualityConstraint::Multipliers(vec![(0, Prescription::Absolute(9.0))]),
            EqualityConstraint::Prescribed(vec![(0, Prescription::Absolute(0.5))]),
            EqualityConstraint::Multipliers(vec![(2, Prescription::Absolute(0.9))]),
        ];
        Assert::default().eq_within_tols(rooted(constraint, None), &Vector::from(SOLUTION))
    }
}

mod block {
    use super::*;
    use crate::math::{
        Matrix, Vector,
        optimize::{RootFindingBlock, SolveStrategy},
        sparse::{CscMatrix, SparseSolver},
    };
    use std::collections::BTreeSet;
    const GLOBAL: usize = 6;
    const BLOCKS: usize = 3;
    const SIZE: usize = 2;
    const LOCAL: usize = BLOCKS * SIZE;
    fn a(i: usize, j: usize) -> Scalar {
        if i == j {
            4.0
        } else if i.abs_diff(j) == 1 {
            -1.0
        } else {
            0.0
        }
    }
    fn m(l: usize, k: usize) -> Scalar {
        [[2.0, 0.5], [0.5, 2.0]][l][k]
    }
    fn b(q: usize, l: usize, k: usize) -> Scalar {
        0.3 + 0.1 * q as Scalar + 0.05 * l as Scalar - 0.02 * k as Scalar
    }
    fn e(q: usize, k: usize, l: usize) -> Scalar {
        0.2 - 0.03 * k as Scalar + 0.04 * l as Scalar + 0.01 * q as Scalar
    }
    fn load(i: usize) -> Scalar {
        1.0 + 0.2 * i as Scalar
    }
    fn window(q: usize, i: usize) -> Option<usize> {
        (q..q + 4).contains(&i).then(|| i - q)
    }
    fn residual_global(u: &Vector, v: &Vector) -> Vector {
        (0..GLOBAL)
            .map(|i| {
                (0..GLOBAL).map(|j| a(i, j) * u[j]).sum::<Scalar>() + 0.05 * u[i].powi(3) - load(i)
                    + (0..BLOCKS)
                        .filter_map(|q| {
                            window(q, i).map(|k| {
                                (0..SIZE)
                                    .map(|l| e(q, k, l) * v[SIZE * q + l])
                                    .sum::<Scalar>()
                            })
                        })
                        .sum::<Scalar>()
            })
            .collect()
    }
    fn residual_local(u: &Vector, v: &Vector) -> Vector {
        (0..LOCAL)
            .map(|i| {
                let (q, l) = (i / SIZE, i % SIZE);
                (0..SIZE).map(|k| m(l, k) * v[SIZE * q + k]).sum::<Scalar>() + 0.1 * v[i].powi(3)
                    - (0..4).map(|k| b(q, l, k) * u[q + k]).sum::<Scalar>()
            })
            .collect()
    }
    fn kuu(i: usize, j: usize, u: &Vector) -> Scalar {
        a(i, j) + if i == j { 0.15 * u[i].powi(2) } else { 0.0 }
    }
    fn kuv(i: usize, j: usize) -> Scalar {
        let (q, l) = (j / SIZE, j % SIZE);
        window(q, i).map_or(0.0, |k| e(q, k, l))
    }
    fn kvu(i: usize, j: usize) -> Scalar {
        let (q, l) = (i / SIZE, i % SIZE);
        window(q, j).map_or(0.0, |k| -b(q, l, k))
    }
    fn kvv(i: usize, j: usize, v: &Vector) -> Scalar {
        if i / SIZE != j / SIZE {
            return 0.0;
        }
        m(i % SIZE, j % SIZE) + if i == j { 0.3 * v[i].powi(2) } else { 0.0 }
    }
    fn patterns() -> [Vec<(usize, usize)>; 4] {
        let mut uu = BTreeSet::new();
        let (mut uv, mut vu, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        (0..BLOCKS).for_each(|q| {
            (q..q + 4).for_each(|i| {
                (q..q + 4).for_each(|j| {
                    uu.insert((i, j));
                });
                (0..SIZE).for_each(|l| {
                    uv.push((i, SIZE * q + l));
                    vu.push((SIZE * q + l, i))
                })
            });
            (0..SIZE).for_each(|l| (0..SIZE).for_each(|k| vv.push((SIZE * q + l, SIZE * q + k))))
        });
        (0..GLOBAL).for_each(|i| {
            uu.insert((i, i));
            if i > 0 {
                uu.insert((i, i - 1));
                uu.insert((i - 1, i));
            }
        });
        [uu.into_iter().collect(), uv, vu, vv]
    }
    fn constraint() -> (CscMatrix, Vector) {
        let mut matrix = CscMatrix::from_pattern(1, GLOBAL, vec![(0, 0)]);
        matrix.fill(|_, _| 1.0);
        (matrix, Vector::from(vec![0.3]))
    }
    fn none() -> (CscMatrix, Vector) {
        (
            CscMatrix::from_pattern(0, LOCAL, Vec::new()),
            Vector::zero(0),
        )
    }
    fn dense(strategy: SolveStrategy) -> Result<(Vector, Vector), OptimizationError> {
        NewtonRaphson::default().root_block(
            |u: &Vector, v: &Vector| Ok(residual_global(u, v)),
            |u: &Vector, v: &Vector| Ok(residual_local(u, v)),
            |u: &Vector, v: &Vector| {
                let build = |height: usize, width: usize, f: &dyn Fn(usize, usize) -> Scalar| {
                    let mut matrix = Matrix::zero(height, width);
                    (0..height).for_each(|i| (0..width).for_each(|j| matrix[i][j] = f(i, j)));
                    matrix
                };
                Ok((
                    build(GLOBAL, GLOBAL, &|i, j| kuu(i, j, u)),
                    build(LOCAL, GLOBAL, &kvu),
                    build(GLOBAL, LOCAL, &kuv),
                    build(LOCAL, LOCAL, &|i, j| kvv(i, j, v)),
                ))
            },
            (Vector::zero(GLOBAL), Vector::zero(LOCAL)),
            constraint(),
            none(),
            None,
            strategy,
        )
    }
    fn sparse(strategy: SolveStrategy) -> Result<(Vector, Vector), OptimizationError> {
        let [uu, uv, vu, vv] = patterns();
        let mut pattern = uu.clone();
        pattern.extend([(GLOBAL, 0), (0, GLOBAL)]);
        let solver = if matches!(strategy, SolveStrategy::Monolithic { elimination: true }) {
            SparseSolver::from_pattern(GLOBAL + 1, pattern, false)
        } else {
            let outer = GLOBAL + 1;
            pattern.extend(uv.iter().map(|&(i, j)| (i, outer + j)));
            pattern.extend(vu.iter().map(|&(i, j)| (outer + i, j)));
            pattern.extend(vv.iter().map(|&(i, j)| (outer + i, outer + j)));
            SparseSolver::from_pattern(outer + LOCAL, pattern, false)
        };
        NewtonRaphson::default().root_block(
            |u: &Vector, v: &Vector| Ok(residual_global(u, v)),
            |u: &Vector, v: &Vector| Ok(residual_local(u, v)),
            |u: &Vector, v: &Vector| {
                let build = |height: usize,
                             width: usize,
                             pattern: &Vec<(usize, usize)>,
                             f: &dyn Fn(usize, usize) -> Scalar| {
                    let mut matrix = CscMatrix::from_pattern(height, width, pattern.clone());
                    matrix.fill(f);
                    matrix
                };
                Ok((
                    build(GLOBAL, GLOBAL, &uu, &|i, j| kuu(i, j, u)),
                    build(LOCAL, GLOBAL, &vu, &kvu),
                    build(GLOBAL, LOCAL, &uv, &kuv),
                    build(LOCAL, LOCAL, &vv, &|i, j| kvv(i, j, v)).with_block_size(SIZE),
                ))
            },
            (Vector::zero(GLOBAL), Vector::zero(LOCAL)),
            constraint(),
            none(),
            Some(solver),
            strategy,
        )
    }
    fn agree(reference: &(Vector, Vector), other: &(Vector, Vector)) -> Result<(), AssertionError> {
        let assert = Assert {
            abs_tol: 1e-10,
            rel_tol: 1e-10,
            ..Default::default()
        };
        assert.eq_within_tols(&reference.0, &other.0)?;
        assert.eq_within_tols(&reference.1, &other.1)
    }
    #[test]
    fn dense_strategies_agree_and_satisfy_the_constraint() -> Result<(), AssertionError> {
        let full = dense(SolveStrategy::Monolithic { elimination: false })?;
        Assert::default().eq_within_tols(full.0[0], &0.3)?;
        assert!(full.1.iter().any(|entry| entry.abs() > 1e-3));
        agree(
            &full,
            &dense(SolveStrategy::Monolithic { elimination: true })?,
        )
    }
    #[test]
    fn sparse_elimination_matches_dense_elimination() -> Result<(), AssertionError> {
        agree(
            &dense(SolveStrategy::Monolithic { elimination: true })?,
            &sparse(SolveStrategy::Monolithic { elimination: true })?,
        )
    }
    #[test]
    fn sparse_elimination_matches_the_sparse_full_system() -> Result<(), AssertionError> {
        agree(
            &sparse(SolveStrategy::Monolithic { elimination: false })?,
            &sparse(SolveStrategy::Monolithic { elimination: true })?,
        )
    }
}

mod verified_minimum {
    use super::*;
    use crate::math::{Matrix, SquareMatrix, Vector, sparse::SparseSolver};
    fn diagonal(entries: [Scalar; 2]) -> impl Fn(&Vector) -> Result<SquareMatrix, String> {
        move |_| Ok(SquareMatrix::from([[entries[0], 0.0], [0.0, entries[1]]]))
    }
    fn solver(sparse: bool, num: usize, pattern: Vec<(usize, usize)>) -> Option<SparseSolver> {
        sparse.then(|| SparseSolver::from_pattern(num, pattern, true))
    }
    fn minimized(
        curvature: [Scalar; 2],
        equality_constraint: EqualityConstraints,
        sparse: Option<SparseSolver>,
    ) -> Result<Vector, OptimizationError> {
        NewtonRaphson::default().minimize(
            |x: &Vector| Ok((curvature[0] * x[0].powi(2) + curvature[1] * x[1].powi(2)) / 2.0),
            |x: &Vector| Ok(Vector::from([curvature[0] * x[0], curvature[1] * x[1]])),
            diagonal(curvature),
            Vector::from([1.0, 1.0]),
            equality_constraint,
            sparse,
        )
    }
    fn diagonal_pattern() -> Vec<(usize, usize)> {
        vec![(0, 0), (1, 1)]
    }
    fn bordered_pattern() -> Vec<(usize, usize)> {
        vec![(0, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1)]
    }
    fn sum_constraint() -> EqualityConstraint {
        let mut matrix = Matrix::zero(1, 2);
        matrix[0][0] = 1.0;
        matrix[0][1] = 1.0;
        EqualityConstraint::Linear(matrix, Vector::from([2.0]))
    }
    fn is_not_minimum(result: Result<Vector, OptimizationError>) -> bool {
        result.is_err_and(|error| error.to_string().contains("not a minimum"))
    }
    const BOTH: [bool; 2] = [true, false];
    #[test]
    fn unconstrained_minimum() {
        BOTH.iter().for_each(|&sparse| {
            let solver = solver(sparse, 2, diagonal_pattern());
            assert!(minimized([1.0, 2.0], vec![EqualityConstraint::None], solver).is_ok())
        })
    }
    #[test]
    fn unconstrained_saddle() {
        BOTH.iter().for_each(|&sparse| {
            let solver = solver(sparse, 2, diagonal_pattern());
            assert!(is_not_minimum(minimized(
                [1.0, -1.0],
                vec![EqualityConstraint::None],
                solver
            )))
        })
    }
    #[test]
    fn unconstrained_maximum() {
        BOTH.iter().for_each(|&sparse| {
            let solver = solver(sparse, 2, diagonal_pattern());
            assert!(is_not_minimum(minimized(
                [-1.0, -2.0],
                vec![EqualityConstraint::None],
                solver
            )))
        })
    }
    #[test]
    fn fixed_saddle() {
        let run = |curvature: [Scalar; 2], sparse: bool| {
            NewtonRaphson::default().minimize(
                |x: &Vector| {
                    Ok(
                        (curvature[0] * x[0].powi(2) + curvature[1] * x[1].powi(2) + x[2].powi(2))
                            / 2.0,
                    )
                },
                |x: &Vector| {
                    Ok(Vector::from([
                        curvature[0] * x[0],
                        curvature[1] * x[1],
                        x[2],
                    ]))
                },
                |_: &Vector| {
                    Ok(SquareMatrix::from([
                        [curvature[0], 0.0, 0.0],
                        [0.0, curvature[1], 0.0],
                        [0.0, 0.0, 1.0],
                    ]))
                },
                Vector::from([1.0, 1.0, 1.0]),
                vec![EqualityConstraint::Fixed(vec![2])],
                solver(sparse, 2, diagonal_pattern()),
            )
        };
        BOTH.iter().for_each(|&sparse| {
            assert!(run([1.0, 2.0], sparse).is_ok());
            assert!(is_not_minimum(run([1.0, -2.0], sparse)))
        })
    }
    #[test]
    fn constrained_minimum_with_indefinite_hessian() {
        BOTH.iter().for_each(|&sparse| {
            let solver = solver(sparse, 3, bordered_pattern());
            assert!(minimized([3.0, -1.0], vec![sum_constraint()], solver).is_ok())
        })
    }
    #[test]
    fn constrained_saddle() {
        BOTH.iter().for_each(|&sparse| {
            let solver = solver(sparse, 3, bordered_pattern());
            assert!(is_not_minimum(minimized(
                [1.0, -3.0],
                vec![sum_constraint()],
                solver
            )))
        })
    }
    #[test]
    fn scalar_minimum_and_maximum() {
        let run = |curvature: Scalar| {
            NewtonRaphson::default().minimize(
                |x: &Scalar| Ok(curvature * x.powi(2) / 2.0),
                |x: &Scalar| Ok(curvature * x),
                |_: &Scalar| Ok(curvature),
                1.0,
                vec![EqualityConstraint::None],
                None,
            )
        };
        assert!(run(1.0).is_ok());
        assert!(run(-1.0).is_err_and(|error| error.to_string().contains("not a minimum")))
    }
}
