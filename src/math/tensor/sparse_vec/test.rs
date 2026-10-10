use crate::math::{
    Current, Hessian, HessianBlock, Quantity, SquareMatrix, TensorBlock2x2, TensorRank1,
    TensorRank2, TensorSparseVec2D, TensorTuple,
};

type Block = TensorBlock2x2<
    TensorRank2<3, Current, Current>,
    TensorRank1<3, Current>,
    TensorRank1<3, Current>,
    Quantity,
>;

fn block(offset: f64) -> Block {
    TensorTuple(
        TensorTuple(
            TensorRank2::from([
                [1.0 + offset, 2.0, 3.0],
                [4.0, 5.0 + offset, 6.0],
                [7.0, 8.0, 9.0 + offset],
            ]),
            TensorRank1::from([10.0, 11.0 + offset, 12.0]),
        ),
        TensorTuple(
            TensorRank1::from([13.0, 14.0, 15.0 + offset]),
            Quantity::new(16.0 + offset),
        ),
    )
}

fn expected(block: &Block, row: usize, column: usize) -> f64 {
    block.entry(row, column)
}

fn container() -> TensorSparseVec2D<Block> {
    let mut container = TensorSparseVec2D::zero(3);
    container[0][1] += block(0.0);
    container[2][2] += block(100.0);
    container[2][2] += block(1.0);
    container
}

#[test]
fn entry() {
    let container = container();
    (0..12).for_each(|row| {
        (0..12).for_each(|column| {
            let (a, b) = (row / 4, column / 4);
            let value = match (a, b) {
                (0, 1) => expected(&block(0.0), row % 4, column % 4),
                (2, 2) => expected(&(block(100.0) + &block(1.0)), row % 4, column % 4),
                _ => 0.0,
            };
            assert_eq!(container.entry(row, column), value)
        })
    })
}

#[test]
fn accumulation_adds_into_the_same_block() {
    let sum = block(100.0) + &block(1.0);
    assert_eq!(sum.0.0[0][0].value(), 1.0 + 100.0 + 1.0 + 1.0)
}

#[test]
fn fill_into() {
    let container = container();
    let mut dense = SquareMatrix::zero(12);
    container.clone().fill_into(&mut dense);
    (0..12).for_each(|row| {
        (0..12).for_each(|column| assert_eq!(dense[row][column], container.entry(row, column)))
    })
}

#[test]
fn retain_from() {
    let container = container();
    let retained = [
        true, false, true, true, false, true, true, true, true, true, false, true,
    ];
    let kept: Vec<usize> = (0..12).filter(|&i| retained[i]).collect();
    let dense = container.clone().retain_from(&retained);
    kept.iter().enumerate().for_each(|(i, &row)| {
        kept.iter()
            .enumerate()
            .for_each(|(j, &column)| assert_eq!(dense[i][j], container.entry(row, column)))
    })
}
