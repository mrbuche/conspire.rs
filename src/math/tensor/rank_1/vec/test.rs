use crate::math::assert::Assert;
use crate::math::{Current, Reference};
use crate::math::{TensorRank0, TensorRank1Vec, TensorVec, Vector, assert::AssertionError};

fn get_array() -> [[TensorRank0; 3]; 2] {
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
}

fn get_tensor_rank_1_vec() -> TensorRank1Vec<3, Current> {
    TensorRank1Vec::from(get_vec_arr())
}

fn get_vec_arr() -> Vec<[TensorRank0; 3]> {
    get_array().to_vec()
}

fn get_vec_vec() -> Vec<Vec<TensorRank0>> {
    get_array().iter().map(|array| array.to_vec()).collect()
}

#[test]
fn from_tensor_rank_1_vec_into_vec_arr() {
    assert_eq!(
        &Vec::<[TensorRank0; 3]>::from(get_tensor_rank_1_vec()),
        &get_vec_arr(),
    )
}

#[test]
fn from_vec_arr_into_tensor_rank_1_vec() -> Result<(), AssertionError> {
    Assert::eq(get_tensor_rank_1_vec(), &get_vec_arr().into())
}

#[test]
fn from_vec_arr_round_trip_preserves_capacity() {
    let mut vec_arr = Vec::with_capacity(9);
    vec_arr.extend_from_slice(&get_array());
    let capacity = vec_arr.capacity();
    assert_eq!(
        Vec::<[TensorRank0; 3]>::from(TensorRank1Vec::<3, Current>::from(vec_arr)).capacity(),
        capacity,
    )
}

fn get_vec_scalar(capacity: usize) -> Vec<TensorRank0> {
    let mut vec = Vec::with_capacity(capacity);
    vec.extend_from_slice(get_array().as_flattened());
    vec
}

#[test]
fn from_configuration_round_trip_preserves_capacity() {
    let mut vec_arr = Vec::with_capacity(9);
    vec_arr.extend_from_slice(&get_array());
    let capacity = vec_arr.capacity();
    let tensor_rank_1_vec = TensorRank1Vec::<3, Reference>::from(vec_arr);
    assert_eq!(
        TensorRank1Vec::<3, Reference>::from(TensorRank1Vec::<3, Current>::from(tensor_rank_1_vec))
            .capacity(),
        capacity,
    )
}

#[test]
fn from_vector_preserves_capacity() {
    let vec = get_vec_scalar(9);
    let capacity = vec.capacity();
    assert!(capacity.is_multiple_of(3));
    assert_eq!(
        TensorRank1Vec::<3, Current>::from(Vector::from(vec)).capacity(),
        capacity / 3,
    )
}

#[test]
fn from_vector_indivisible_capacity() -> Result<(), AssertionError> {
    let vec = get_vec_scalar(7);
    assert!(!vec.capacity().is_multiple_of(3));
    Assert::eq(
        get_tensor_rank_1_vec(),
        &TensorRank1Vec::<3, Current>::from(Vector::from(vec)),
    )
}

#[test]
fn from_tensor_rank_1_vec_into_vec_vec() {
    assert_eq!(
        &Vec::<Vec::<TensorRank0>>::from(get_tensor_rank_1_vec()),
        &get_vec_vec(),
    );
}

#[test]
fn from_vec_vec_into_tensor_rank_1_vec() -> Result<(), AssertionError> {
    Assert::eq(get_tensor_rank_1_vec(), &get_vec_vec().into())
}
