use super::{
    super::super::test::{TestError, assert_eq},
    TensorRank0, TensorRank1Vec, TensorVec,
};

fn get_array() -> [[TensorRank0; 3]; 2] {
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
}

fn get_tensor_rank_1_vec() -> TensorRank1Vec<3, 1> {
    TensorRank1Vec::new(&get_array())
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
fn from_vec_arr_into_tensor_rank_1_vec() -> Result<(), TestError> {
    assert_eq(&get_tensor_rank_1_vec(), &get_vec_arr().into())
}

#[test]
fn from_tensor_rank_1_vec_into_vec_vec() {
    assert_eq!(
        &Vec::<Vec::<TensorRank0>>::from(get_tensor_rank_1_vec()),
        &get_vec_vec(),
    );
}

#[test]
fn from_vec_vec_into_tensor_rank_1_vec() -> Result<(), TestError> {
    assert_eq(&get_tensor_rank_1_vec(), &get_vec_vec().into())
}
