use super::{SparseSolver, Vector};
use crate::math::test::{TestError, assert_eq_within_tols};

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

fn residual(source: impl Fn(usize, usize) -> f64, x: &Vector, b: &Vector) -> Result<(), TestError> {
    let mut product = Vector::zero(N);
    pattern()
        .into_iter()
        .for_each(|(i, j)| product[i] += source(i, j) * x[j]);
    assert_eq_within_tols(&product, b)
}

#[test]
fn solve_factor_then_refactor() -> Result<(), TestError> {
    let solver = SparseSolver::from_pattern(N, pattern(), false);
    let b: Vector = (0..N).map(|i| (i % 13) as f64 - 6.0).collect();
    residual(values(1.0), &solver.solve(values(1.0), &b)?, &b)?;
    residual(values(-2.0), &solver.solve(values(-2.0), &b)?, &b)
}

#[test]
fn clones_share_factorization() -> Result<(), TestError> {
    let solver = SparseSolver::from_pattern(N, pattern(), false);
    let b: Vector = (0..N).map(|i| (i % 13) as f64 - 6.0).collect();
    residual(values(1.0), &solver.clone().solve(values(1.0), &b)?, &b)?;
    assert!(solver.lu.borrow().is_some());
    residual(values(3.0), &solver.clone().solve(values(3.0), &b)?, &b)
}

#[test]
fn recovers_from_degraded_pivot() -> Result<(), TestError> {
    let solver = SparseSolver::from_pattern(2, vec![(0, 0), (0, 1), (1, 0), (1, 1)], false);
    let b = Vector::from([1.0, 1.0]);
    let x = solver.solve(|i, j| ((2 * i + j) % 3) as f64 + 1.0, &b)?;
    assert_eq_within_tols(&Vector::from([x[0] + 2.0 * x[1], 3.0 * x[0] + x[1]]), &b)?;
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
    assert_eq_within_tols(&Vector::from([x[0] + 2.0 * x[1], 2.0 * x[1]]), &b)
}

#[test]
fn symmetric_uses_ldl() -> Result<(), TestError> {
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
    assert_eq_within_tols(&residual, &b)?;
    assert!(solver.ldl.borrow().is_some());
    let x = solver.solve(source(-2.0), &b)?;
    let mut residual = Vector::zero(n + 4);
    solver
        .pattern()
        .iter()
        .for_each(|&(i, j)| residual[i] += source(-2.0)(i, j) * x[j]);
    assert_eq_within_tols(&residual, &b)
}

#[test]
fn asymmetric_falls_back_to_lu() -> Result<(), TestError> {
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
    assert_eq_within_tols(&residual, &b)
}
