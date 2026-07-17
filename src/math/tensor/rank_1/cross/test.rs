use super::CrossProduct;
use crate::math::assert::Assert;
use crate::math::{TensorRank1, assert::AssertionError};

fn get_tensor_rank_1_b() -> TensorRank1<3, 1> {
    TensorRank1::from([7.0, 2.0, 3.0])
}

fn get_tensor_rank_1_c() -> TensorRank1<3, 1> {
    TensorRank1::from([4.0, 5.0, 6.0])
}

fn get_tensor_rank_1_b_cross_c() -> TensorRank1<3, 1> {
    TensorRank1::from([-3.0, -30.0, 27.0])
}

#[test]
fn cross() -> Result<(), AssertionError> {
    Assert::eq(
        get_tensor_rank_1_b().cross(&get_tensor_rank_1_c()),
        &get_tensor_rank_1_b_cross_c(),
    )
}
