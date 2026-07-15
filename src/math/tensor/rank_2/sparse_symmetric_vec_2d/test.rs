use super::{TensorRank2, TensorRank2SparseVec2DSymmetric};
use crate::math::{Hessian, HessianAccumulate, Rank2, SquareMatrix};

fn block(value: f64) -> TensorRank2<2, 1, 1> {
    TensorRank2::from([[value, 2.0 * value], [3.0 * value, 4.0 * value]])
}

fn accumulator() -> TensorRank2SparseVec2DSymmetric<2, 1, 1> {
    let mut stiffnesses = TensorRank2SparseVec2DSymmetric::<2, 1, 1>::zero(3);
    stiffnesses.accumulate(0, 2, block(1.0));
    stiffnesses.accumulate(0, 0, block(2.0));
    stiffnesses.accumulate(1, 1, block(5.0));
    stiffnesses
}

fn dense() -> SquareMatrix {
    let mut square_matrix = SquareMatrix::zero(6);
    accumulator().fill_into(&mut square_matrix);
    square_matrix
}

#[test]
fn accumulate_canonicalizes_out_of_order_pairs() {
    let mut out_of_order = TensorRank2SparseVec2DSymmetric::<2, 1, 1>::zero(3);
    out_of_order.accumulate(2, 0, block(1.0));
    let mut in_order = TensorRank2SparseVec2DSymmetric::<2, 1, 1>::zero(3);
    in_order.accumulate(0, 2, block(1.0).transpose());
    (0..6).for_each(|p| {
        (0..6).for_each(|q| {
            assert_eq!(out_of_order.entry(p, q), in_order.entry(p, q));
        })
    });
    assert_eq!(out_of_order.entry(4, 0), block(1.0)[0][0]);
    assert_eq!(out_of_order.entry(4, 1), block(1.0)[0][1]);
    assert_eq!(out_of_order.entry(5, 0), block(1.0)[1][0]);
    assert_eq!(out_of_order.entry(5, 1), block(1.0)[1][1]);
}

#[test]
fn entry_mirrors_off_diagonal_node_pairs() {
    // Diagonal (same-node) blocks are a caller contract, not enforced here;
    // only cross-node entries are guaranteed transpose-mirrored by construction.
    let stiffnesses = accumulator();
    (0..6).for_each(|p| {
        (0..6).for_each(|q| {
            if p / 2 != q / 2 {
                assert_eq!(stiffnesses.entry(p, q), stiffnesses.entry(q, p));
            }
        })
    });
}

#[test]
fn entry_reads_canonical_and_missing_positions() {
    let stiffnesses = accumulator();
    assert_eq!(stiffnesses.entry(0, 0), 2.0);
    assert_eq!(stiffnesses.entry(0, 4), 1.0);
    assert_eq!(stiffnesses.entry(4, 0), 1.0);
    assert_eq!(stiffnesses.entry(1, 4), 3.0);
    assert_eq!(stiffnesses.entry(4, 1), 3.0);
    assert_eq!(stiffnesses.entry(0, 2), 0.0);
    assert_eq!(stiffnesses.entry(2, 0), 0.0);
}

#[test]
fn fill_into_mirrors_off_diagonal_blocks() {
    let square_matrix = dense();
    assert_eq!(square_matrix[0][0], 2.0);
    assert_eq!(square_matrix[0][4], 1.0);
    assert_eq!(square_matrix[4][0], 1.0);
    assert_eq!(square_matrix[1][5], 4.0);
    assert_eq!(square_matrix[5][1], 4.0);
    assert_eq!(square_matrix[2][2], 5.0);
    assert_eq!(square_matrix[0][2], 0.0);
}

#[test]
fn retain_from_filters_and_mirrors() {
    let retained = [true, false, true, true, true, false];
    let square_matrix = accumulator().retain_from(&retained);
    let full = dense();
    let kept: Vec<usize> = (0..6).filter(|&p| retained[p]).collect();
    kept.iter().enumerate().for_each(|(p, &full_p)| {
        kept.iter()
            .enumerate()
            .for_each(|(q, &full_q)| assert_eq!(square_matrix[p][q], full[full_p][full_q]))
    });
}

#[test]
fn add_and_sub_operate_entrywise() {
    let mut other = TensorRank2SparseVec2DSymmetric::<2, 1, 1>::zero(3);
    other.accumulate(0, 0, block(7.0));
    let sum = accumulator() + other.clone();
    assert_eq!(sum.entry(0, 0), 9.0);
    assert_eq!(sum.entry(2, 2), 5.0);
    let difference = accumulator() - other;
    assert_eq!(difference.entry(0, 0), -5.0);
}

#[test]
fn mul_and_div_scale_all_entries() {
    let scaled = accumulator() * 2.0;
    assert_eq!(scaled.entry(0, 0), 4.0);
    assert_eq!(scaled.entry(0, 4), 2.0);
    let divided = scaled / 2.0;
    assert_eq!(divided.entry(0, 0), 2.0);
}
