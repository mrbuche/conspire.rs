use crate::math::Current;
use crate::math::{TensorError, TensorRank2};

#[test]
fn powm_non_positive_diagonal_entry_errors() {
    assert_eq!(
        TensorRank2::<3, Current, Current>::from([
            [2.0, 0.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.0, 1.5],
        ])
        .powm(0.5),
        Err(TensorError::NotPositiveDefinite)
    )
}
