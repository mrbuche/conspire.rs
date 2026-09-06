//! Shared assertions for the autodiff model cross-checks.

use crate::{
    constitutive::ConstitutiveError,
    math::{TensorRank2, TensorRank4},
    units::Stress,
};

pub(crate) fn ok<T>(result: Result<T, ConstitutiveError>) -> T {
    match result {
        Ok(value) => value,
        Err(_) => panic!("evaluation failed"),
    }
}

pub(crate) fn assert_close_2<I, J>(
    ad: &TensorRank2<3, I, J, Stress>,
    hand: &TensorRank2<3, I, J, Stress>,
    tol: f64,
) {
    for i in 0..3 {
        for j in 0..3 {
            let (a, b) = (ad[i][j].value(), hand[i][j].value());
            assert!(
                (a - b).abs() <= tol * (1.0 + b.abs()),
                "[{i}][{j}]: {a} vs {b}"
            );
        }
    }
}

pub(crate) fn assert_close_4<I, J, K, L>(
    ad: &TensorRank4<3, I, J, K, L, Stress>,
    hand: &TensorRank4<3, I, J, K, L, Stress>,
    tol: f64,
) {
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    let (a, b) = (ad[i][j][k][l].value(), hand[i][j][k][l].value());
                    assert!(
                        (a - b).abs() <= tol * (1.0 + b.abs()),
                        "[{i}][{j}][{k}][{l}]: {a} vs {b}"
                    );
                }
            }
        }
    }
}
