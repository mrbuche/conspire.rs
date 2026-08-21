use super::{Scalar, SparseSolver, Vector};
use crate::math::assert::Assert;
use crate::math::assert::AssertionError;
use crate::math::matrix::square::SquareMatrix;

const N: usize = 100;

fn pattern() -> Vec<(usize, usize)> {
    let mut pattern: Vec<(usize, usize)> = (0..N).map(|i| (i, i)).collect();
    (0..N).for_each(|i| {
        pattern.push((i, (3 * i + 1) % N));
        pattern.push(((5 * i + 2) % N, i));
    });
    pattern
}

fn values(scale: f64) -> impl Fn(usize, usize) -> f64 {
    move |i, j| {
        if i == j {
            8.0
        } else {
            scale * (((i * 7 + j * 3) % 5) as f64 / 5.0 - 0.4)
        }
    }
}

fn residual(
    source: impl Fn(usize, usize) -> f64,
    x: &Vector,
    b: &Vector,
) -> Result<(), AssertionError> {
    let mut product = Vector::zero(N);
    pattern()
        .into_iter()
        .for_each(|(i, j)| product[i] += source(i, j) * x[j]);
    Assert::default().eq_within_tols(&product, b)
}

#[test]
fn solve_factor_then_refactor() -> Result<(), AssertionError> {
    let solver = SparseSolver::from_pattern(N, pattern(), false);
    let b: Vector = (0..N).map(|i| (i % 13) as f64 - 6.0).collect();
    residual(values(1.0), &solver.solve(values(1.0), &b)?, &b)?;
    residual(values(-2.0), &solver.solve(values(-2.0), &b)?, &b)
}

#[test]
fn clones_share_factorization() -> Result<(), AssertionError> {
    let solver = SparseSolver::from_pattern(N, pattern(), false);
    let b: Vector = (0..N).map(|i| (i % 13) as f64 - 6.0).collect();
    residual(values(1.0), &solver.clone().solve(values(1.0), &b)?, &b)?;
    assert!(solver.lu.borrow().is_some());
    residual(values(3.0), &solver.clone().solve(values(3.0), &b)?, &b)
}

#[test]
fn recovers_from_degraded_pivot() -> Result<(), AssertionError> {
    let solver = SparseSolver::from_pattern(2, vec![(0, 0), (0, 1), (1, 0), (1, 1)], false);
    let b = Vector::from([1.0, 1.0]);
    let x = solver.solve(|i, j| ((2 * i + j) % 3) as f64 + 1.0, &b)?;
    Assert::default().eq_within_tols(Vector::from([x[0] + 2.0 * x[1], 3.0 * x[0] + x[1]]), &b)?;
    let x = solver.solve(
        |i, j| {
            if (i, j) == (1, 0) {
                0.0
            } else {
                j as f64 + 1.0
            }
        },
        &b,
    )?;
    Assert::default().eq_within_tols(Vector::from([x[0] + 2.0 * x[1], 2.0 * x[1]]), &b)
}

#[test]
fn symmetric_uses_ldl() -> Result<(), AssertionError> {
    let n = 30;
    let mut pattern: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
    (0..n - 3).for_each(|i| {
        pattern.push((i, i + 3));
        pattern.push((i + 3, i));
    });
    (0..4).for_each(|c| {
        pattern.push((n + c, 7 * c));
        pattern.push((7 * c, n + c));
    });
    let solver = SparseSolver::from_pattern(n + 4, pattern, true);
    let b: Vector = (0..n + 4).map(|i| (i % 7) as f64 - 3.0).collect();
    let source = |scale: f64| {
        move |i: usize, j: usize| {
            if i == j {
                12.0
            } else if i >= n || j >= n {
                2.0
            } else {
                scale * ((i.min(j) % 3) as f64 - 1.0)
            }
        }
    };
    let x = solver.solve(source(1.0), &b)?;
    let mut residual = Vector::zero(n + 4);
    solver
        .pattern()
        .iter()
        .for_each(|&(i, j)| residual[i] += source(1.0)(i, j) * x[j]);
    Assert::default().eq_within_tols(&residual, &b)?;
    assert!(solver.ldl.borrow().is_some());
    let x = solver.solve(source(-2.0), &b)?;
    let mut residual = Vector::zero(n + 4);
    solver
        .pattern()
        .iter()
        .for_each(|&(i, j)| residual[i] += source(-2.0)(i, j) * x[j]);
    Assert::default().eq_within_tols(&residual, &b)
}

#[test]
fn asymmetric_falls_back_to_lu() -> Result<(), AssertionError> {
    let n = 20;
    let mut pattern: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
    (0..n - 3).for_each(|i| {
        pattern.push((i, i + 3));
        pattern.push((i + 3, i));
    });
    let solver = SparseSolver::from_pattern(n, pattern, false);
    let b: Vector = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
    let source = |i: usize, j: usize| {
        if i == j {
            12.0
        } else {
            (i % 3) as f64 - (j % 2) as f64
        }
    };
    let x = solver.solve(source, &b)?;
    assert!(solver.ldl.borrow().is_none());
    assert!(solver.lu.borrow().is_some());
    let mut residual = Vector::zero(n);
    solver
        .pattern()
        .iter()
        .for_each(|&(i, j)| residual[i] += source(i, j) * x[j]);
    Assert::default().eq_within_tols(&residual, &b)
}

#[test]
fn degraded_pivot_keeps_the_ldl_for_the_next_solve() -> Result<(), AssertionError> {
    let pattern = vec![(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)];
    let solver = SparseSolver::from_pattern(3, pattern.clone(), true);
    let b = Vector::from([1.0, 1.0, 1.0]);
    let good = |i: usize, j: usize| if i == j { 4.0 } else { -1.0 };
    let degraded = |i: usize, j: usize| {
        if i == j {
            if i == 0 { 0.0 } else { -2.0 }
        } else {
            -2.0
        }
    };
    let check = |source: &dyn Fn(usize, usize) -> Scalar, x: &Vector| {
        let mut residual = Vector::zero(3);
        pattern
            .iter()
            .for_each(|&(i, j)| residual[i] += source(i, j) * x[j]);
        Assert::default().eq_within_tols(&residual, &b)
    };
    check(&good, &solver.solve(good, &b)?)?;
    assert!(solver.ldl.borrow().is_some());
    check(&degraded, &solver.solve(degraded, &b)?)?;
    assert!(solver.ldl.borrow().is_some());
    check(&good, &solver.solve(good, &b)?)?;
    assert!(solver.ldl.borrow().is_some());
    Ok(())
}

/// The inertia a sparse and a dense factorization each report for the same
/// matrix, `structural` naming the positions an assembly would leave in the
/// pattern even though their values happen to be zero.
fn inertia_of(
    dense: &[Vec<Scalar>],
    structural: &[(usize, usize)],
) -> ((usize, usize, usize), (usize, usize, usize)) {
    let n = dense.len();
    let mut pattern: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .filter(|&(i, j)| dense[i][j] != 0.0)
        .collect();
    pattern.extend_from_slice(structural);
    let solver = SparseSolver::from_pattern(n, pattern, true);
    let b = Vector::zero(n);
    let sparse = solver.solve_ldl(|i, j| dense[i][j], &b).unwrap().1;
    let mut square = SquareMatrix::zero(n);
    (0..n).for_each(|i| (0..n).for_each(|j| square[i][j] = dense[i][j]));
    (sparse, square.factorize_ldl().unwrap().inertia())
}

#[test]
fn inertia_matches_the_dense_factorization() {
    let positive_definite = vec![
        vec![4.0, -1.0, 0.0],
        vec![-1.0, 4.0, -1.0],
        vec![0.0, -1.0, 4.0],
    ];
    let negative_definite: Vec<Vec<Scalar>> = positive_definite
        .iter()
        .map(|row| row.iter().map(|value| -value).collect())
        .collect();
    //
    // The constraint row carries no diagonal entry at all, so the maximum
    // transversal has to pair it with a variable row: this is where the
    // two-by-two blocks a bordered system produces get counted.
    //
    let bordered = vec![
        vec![4.0, -1.0, 1.0],
        vec![-1.0, 4.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ];
    let bordered_indefinite = vec![
        vec![-3.0, -1.0, 1.0],
        vec![-1.0, 4.0, 2.0],
        vec![1.0, 2.0, 0.0],
    ];
    //
    // Fill gives the paired row a diagonal of its own, so this one carries a
    // block with a positive determinant: two eigenvalues of the same sign,
    // which the pairing rule of a Bunch-Kaufman factorization could not produce.
    //
    let positive_pair = vec![
        vec![-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0],
        vec![1.0, -1.0, 4.0, 0.0, 0.0, 0.0, -3.0, 0.0],
        vec![0.0, 4.0, 0.0, -3.0, 0.0, 0.0, 3.0, -2.0],
        vec![0.0, 0.0, -3.0, -3.0, -2.0, 0.0, 0.0, -2.0],
        vec![0.0, 0.0, 0.0, -2.0, -1.0, -3.0, 0.0, 2.0],
        vec![0.0, 0.0, 0.0, 0.0, -3.0, -1.0, 0.0, 0.0],
        vec![-2.0, -3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, -2.0, -2.0, 2.0, 0.0, 0.0, 0.0],
    ];
    //
    // Negating a matrix negates D and leaves L and the structural pairing
    // alone, so the same block comes back definite the other way: determinant
    // still positive, trace now negative.
    //
    let negative_pair: Vec<Vec<Scalar>> = positive_pair
        .iter()
        .map(|row| row.iter().map(|value| -value).collect())
        .collect();
    [
        (positive_definite, vec![], (3, 0, 0)),
        (negative_definite, vec![], (0, 3, 0)),
        (bordered, vec![], (2, 1, 0)),
        (bordered_indefinite, vec![], (1, 2, 0)),
        (positive_pair, vec![(2, 2)], (4, 4, 0)),
        (negative_pair, vec![(2, 2)], (4, 4, 0)),
    ]
    .into_iter()
    .for_each(|(dense, structural, expected)| {
        let (sparse, square) = inertia_of(&dense, &structural);
        assert_eq!(sparse, square);
        assert_eq!(sparse, expected);
    })
}

#[test]
fn solve_ldl_refuses_where_solve_recovers() {
    let pattern = vec![(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)];
    let solver = SparseSolver::from_pattern(3, pattern, true);
    let b = Vector::from([1.0, 1.0, 1.0]);
    let degraded = |i: usize, j: usize| {
        if i == j {
            if i == 0 { 0.0 } else { -2.0 }
        } else {
            -2.0
        }
    };
    //
    // The very matrix the LU fallback rescues: an inertia-gated caller must be
    // told the LDLᵀ broke down rather than handed another factorization's answer.
    //
    assert!(solver.solve(degraded, &b).is_ok());
    assert!(solver.solve_ldl(degraded, &b).is_err())
}

#[test]
fn solve_ldl_refuses_an_asymmetric_solver() {
    let solver = SparseSolver::from_pattern(2, vec![(0, 0), (0, 1), (1, 0), (1, 1)], false);
    let b = Vector::from([1.0, 1.0]);
    assert!(
        solver
            .solve_ldl(|i, j| ((2 * i + j) % 3) as f64 + 1.0, &b)
            .is_err()
    )
}
