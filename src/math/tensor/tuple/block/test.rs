use crate::math::{
    Current, HessianBlock, Matrix, Quantity, TensorBlock2x2, TensorRank1, TensorRank2, TensorTuple,
};

fn block() -> TensorBlock2x2<
    TensorRank2<3, Current, Current>,
    TensorRank1<3, Current>,
    TensorRank1<3, Current>,
    Quantity,
> {
    TensorTuple(
        TensorTuple(
            TensorRank2::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            TensorRank1::from([10.0, 11.0, 12.0]),
        ),
        TensorTuple(TensorRank1::from([13.0, 14.0, 15.0]), Quantity::new(16.0)),
    )
}

const EXPECTED: [[f64; 4]; 4] = [
    [1.0, 2.0, 3.0, 10.0],
    [4.0, 5.0, 6.0, 11.0],
    [7.0, 8.0, 9.0, 12.0],
    [13.0, 14.0, 15.0, 16.0],
];

#[test]
fn dimensions() {
    assert_eq!((block().height(), block().width()), (4, 4))
}

#[test]
fn entry() {
    (0..4).for_each(|i| (0..4).for_each(|j| assert_eq!(block().entry(i, j), EXPECTED[i][j])))
}

#[test]
fn fill_into_block() {
    let mut matrix = Matrix::zero(6, 6);
    block().fill_into_block(&mut matrix, 1, 2);
    (0..6).for_each(|i| {
        (0..6).for_each(|j| {
            let expected = if (1..5).contains(&i) && (2..6).contains(&j) {
                EXPECTED[i - 1][j - 2]
            } else {
                0.0
            };
            assert_eq!(matrix[i][j], expected)
        })
    })
}
