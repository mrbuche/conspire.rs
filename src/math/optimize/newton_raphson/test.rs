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

mod block {
    use super::*;
    use crate::math::{
        Matrix, Vector,
        optimize::{FirstOrderRootFindingBlock, SolveStrategy},
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
