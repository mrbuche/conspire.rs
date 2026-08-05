use super::{TensorRank2, TensorRank2Vec2D};
use crate::math::{
    Hessian, SquareMatrix, Vector,
    assert::{Assert, AssertionError},
};

fn block(value: f64) -> TensorRank2<2, 1, 1> {
    TensorRank2::from([[value, 2.0 * value], [3.0 * value, 4.0 * value]])
}

fn blocks() -> TensorRank2Vec2D<2, 1, 1> {
    (0..3)
        .map(|a| (0..3).map(|b| block((1 + a * 3 + b) as f64)).collect())
        .collect()
}

fn dense() -> SquareMatrix {
    let mut square_matrix = SquareMatrix::zero(6);
    blocks().fill_into(&mut square_matrix);
    square_matrix
}

#[test]
fn quadratic_form_matches_dense() -> Result<(), AssertionError> {
    let vector = Vector::from([1.0, -2.0, 3.0, 0.5, -1.5, 2.0]);
    Assert::default().eq_within_tols(
        blocks().quadratic_form(&vector),
        &(&vector * (dense() * &vector)),
    )
}
