use super::{Tensor, TensorArray, TensorRank0, TensorRank0List};

fn get_array() -> [TensorRank0; 4] {
    [1.0, 2.0, 3.0, 4.0]
}

fn get_tensor_rank_0_list() -> TensorRank0List<4> {
    TensorRank0List::new(get_array())
}

fn get_other_tensor_rank_0_list() -> TensorRank0List<4> {
    TensorRank0List::new([5.0, 6.0, 7.0, 8.0])
}

#[test]
fn from_iter() {
    let into_iterator = (0..8).map(|x| x as TensorRank0);
    let tensor_rank_0_list = TensorRank0List::<8>::from_iter(into_iterator.clone());
    tensor_rank_0_list
        .iter()
        .zip(into_iterator)
        .for_each(|(tensor_rank_0_list_i, value_i)| assert_eq!(tensor_rank_0_list_i, &value_i));
}

#[test]
fn iter() {
    get_tensor_rank_0_list()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_0_list_i, array_i)| assert_eq!(tensor_rank_0_list_i, array_i));
}

#[test]
fn iter_mut() {
    get_tensor_rank_0_list()
        .iter_mut()
        .zip(get_array().iter_mut())
        .for_each(|(tensor_rank_0_list_i, array_i)| assert_eq!(tensor_rank_0_list_i, array_i));
}

#[test]
fn mul_tensor_rank_1_list_to_self() {
    assert_eq!(
        get_tensor_rank_0_list() * get_other_tensor_rank_0_list(),
        70.0
    )
}

#[test]
fn mul_tensor_rank_1_list_ref_to_self() {
    assert_eq!(
        get_tensor_rank_0_list() * &get_other_tensor_rank_0_list(),
        70.0
    )
}

#[test]
fn mul_tensor_rank_1_list_to_self_ref() {
    assert_eq!(
        &get_tensor_rank_0_list() * get_other_tensor_rank_0_list(),
        70.0
    )
}

#[test]
fn mul_tensor_rank_1_list_ref_to_self_ref() {
    assert_eq!(
        &get_tensor_rank_0_list() * &get_other_tensor_rank_0_list(),
        70.0
    )
}

#[test]
fn new() {
    get_tensor_rank_0_list()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_0_list_i, array_i)| assert_eq!(tensor_rank_0_list_i, array_i));
}

#[test]
fn zero() {
    TensorRank0List::<8>::zero()
        .iter()
        .for_each(|tensor_rank_0_list_i| assert_eq!(tensor_rank_0_list_i, &0.0));
}
