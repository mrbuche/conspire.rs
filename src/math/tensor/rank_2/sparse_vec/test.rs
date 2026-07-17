use super::{TensorRank2, TensorRank2SparseVec};
use crate::math::Tensor;

fn block(value: f64) -> TensorRank2<2, 1, 1> {
    TensorRank2::from([[value, 2.0 * value], [3.0 * value, 4.0 * value]])
}

fn sparse_vec() -> TensorRank2SparseVec<2, 1, 1> {
    let mut vec = TensorRank2SparseVec::default();
    vec[2] += block(1.0);
    vec[0] += block(2.0);
    vec[0] += block(3.0);
    vec
}

#[test]
fn insertion_sums_and_sorts() {
    let vec = sparse_vec();
    assert_eq!(vec[0][0][0], 5.0);
    assert_eq!(vec[2][0][0], 1.0);
    let columns: Vec<usize> = vec.entries().map(|(column, _)| column).collect();
    assert_eq!(columns, [0, 2]);
}

#[test]
#[should_panic(expected = "Entry (1) not present.")]
fn indexing_missing_entry_panics() {
    let vec = sparse_vec();
    let _ = &vec[1];
}

#[test]
fn tensor_len_and_size() {
    let vec = sparse_vec();
    assert_eq!(vec.len(), 2);
    assert_eq!(vec.size(), 2 * 2 * 2);
}

#[test]
fn from_iterator_indexes_sequentially() {
    let vec: TensorRank2SparseVec<2, 1, 1> =
        [block(1.0), block(2.0), block(3.0)].into_iter().collect();
    assert_eq!(vec[0][0][0], 1.0);
    assert_eq!(vec[1][0][0], 2.0);
    assert_eq!(vec[2][0][0], 3.0);
}

#[test]
fn add_and_sub_merge_disjoint_and_overlapping_entries() {
    let mut other = TensorRank2SparseVec::<2, 1, 1>::default();
    other[0] += block(7.0);
    other[1] += block(5.0);
    let sum = sparse_vec() + other.clone();
    assert_eq!(sum[0][0][0], 12.0);
    assert_eq!(sum[1][0][0], 5.0);
    assert_eq!(sum[2][0][0], 1.0);
    let difference = sparse_vec() - other;
    assert_eq!(difference[0][0][0], -2.0);
    assert_eq!(difference[1][0][0], -5.0);
    assert_eq!(difference[2][0][0], 1.0);
}

#[test]
fn add_assign_and_sub_assign_by_reference() {
    let mut vec = sparse_vec();
    let other = sparse_vec();
    vec += &other;
    assert_eq!(vec[0][0][0], 10.0);
    vec -= &other;
    assert_eq!(vec[0][0][0], 5.0);
}

#[test]
fn mul_and_div_scale_all_entries() {
    let scaled = sparse_vec() * 2.0;
    assert_eq!(scaled[0][0][0], 10.0);
    assert_eq!(scaled[2][0][0], 2.0);
    let divided = scaled / 2.0;
    assert_eq!(divided[0][0][0], 5.0);
    assert_eq!(divided[2][0][0], 1.0);
}

#[test]
fn sum_folds_over_iterator() {
    let total: TensorRank2SparseVec<2, 1, 1> = vec![sparse_vec(), sparse_vec()].into_iter().sum();
    assert_eq!(total[0][0][0], 10.0);
    assert_eq!(total[2][0][0], 2.0);
}
