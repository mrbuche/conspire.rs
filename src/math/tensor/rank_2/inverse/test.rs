use crate::math::{
    Rank2, Tensor, TensorArray, TensorRank0, TensorRank2,
    test::{TestError, assert_eq, assert_eq_within_tols},
};
use std::cmp::Ordering;

fn get_array_dim_2() -> [[TensorRank0; 2]; 2] {
    [[1.0, 2.0], [3.0, 4.0]]
}

fn get_array_dim_3() -> [[TensorRank0; 3]; 3] {
    [[1.0, 4.0, 6.0], [7.0, 2.0, 5.0], [9.0, 8.0, 3.0]]
}

fn get_array_dim_4() -> [[TensorRank0; 4]; 4] {
    [
        [1.0, 4.0, 6.0, 6.0],
        [1.0, 5.0, 1.0, 0.0],
        [1.0, 3.0, 5.0, 0.0],
        [1.0, 4.0, 6.0, 0.0],
    ]
}

fn get_array_dim_9() -> [[TensorRank0; 9]; 9] {
    [
        [2.0, 2.0, 4.0, 0.0, 0.0, 1.0, 1.0, 3.0, 3.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 4.0, 2.0, 1.0],
        [3.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 4.0, 2.0],
        [4.0, 4.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 4.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 1.0],
        [4.0, 2.0, 3.0, 4.0, 2.0, 4.0, 3.0, 0.0, 4.0],
        [1.0, 3.0, 2.0, 0.0, 0.0, 0.0, 2.0, 4.0, 2.0],
        [2.0, 2.0, 2.0, 4.0, 1.0, 2.0, 4.0, 2.0, 2.0],
        [1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 4.0, 2.0, 1.0],
    ]
}

fn get_tensor_rank_2_dim_2() -> TensorRank2<2, 1, 1> {
    TensorRank2::from(get_array_dim_2())
}

fn get_tensor_rank_2_dim_3() -> TensorRank2<3, 1, 1> {
    TensorRank2::from(get_array_dim_3())
}

fn get_tensor_rank_2_dim_4() -> TensorRank2<4, 1, 1> {
    TensorRank2::from(get_array_dim_4())
}

fn get_tensor_rank_2_dim_9() -> TensorRank2<9, 1, 1> {
    TensorRank2::from(get_array_dim_9())
}

#[test]
fn determinant_dim_2() -> Result<(), TestError> {
    assert_eq(&get_tensor_rank_2_dim_2().determinant(), &-2.0)
}

#[test]
fn determinant_dim_3() -> Result<(), TestError> {
    assert_eq(&get_tensor_rank_2_dim_3().determinant(), &290.0)
}

#[test]
fn determinant_dim_4() -> Result<(), TestError> {
    assert_eq_within_tols(&get_tensor_rank_2_dim_4().determinant(), &36.0)
}

#[test]
fn determinant_dim_9() -> Result<(), TestError> {
    assert_eq_within_tols(&get_tensor_rank_2_dim_9().determinant(), &5297.0)
}

#[test]
fn inverse_dim_2() -> Result<(), TestError> {
    assert_eq_within_tols(
        &(get_tensor_rank_2_dim_2() * get_tensor_rank_2_dim_2().inverse()),
        &TensorRank2::identity(),
    )
}

#[test]
fn inverse_dim_3() -> Result<(), TestError> {
    assert_eq_within_tols(
        &(get_tensor_rank_2_dim_3() * get_tensor_rank_2_dim_3().inverse()),
        &TensorRank2::identity(),
    )
}

#[test]
fn inverse_dim_4() -> Result<(), TestError> {
    assert_eq_within_tols(
        &(get_tensor_rank_2_dim_4() * get_tensor_rank_2_dim_4().inverse()),
        &TensorRank2::identity(),
    )
}

#[test]
fn inverse_dim_9() -> Result<(), TestError> {
    assert_eq_within_tols(
        &(get_tensor_rank_2_dim_9() * get_tensor_rank_2_dim_9().inverse()),
        &TensorRank2::identity(),
    )
}

#[test]
fn inverse_and_determinant_dim_2() -> Result<(), TestError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_2();
    let (inverse, determinant) = tensor_rank_2.inverse_and_determinant();
    assert_eq(&determinant, &tensor_rank_2.determinant())?;
    assert_eq(&inverse, &tensor_rank_2.inverse())
}

#[test]
fn inverse_and_determinant_dim_3() -> Result<(), TestError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_3();
    let (inverse, determinant) = tensor_rank_2.inverse_and_determinant();
    assert_eq(&determinant, &tensor_rank_2.determinant())?;
    assert_eq(&inverse, &tensor_rank_2.inverse())
}

#[test]
fn inverse_and_determinant_dim_4() -> Result<(), TestError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_4();
    let (inverse, determinant) = tensor_rank_2.inverse_and_determinant();
    assert_eq(&determinant, &tensor_rank_2.determinant())?;
    assert_eq(&inverse, &tensor_rank_2.inverse())
}

#[test]
fn inverse_and_determinant_dim_9() -> Result<(), TestError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_9();
    let (inverse, determinant) = tensor_rank_2.inverse_and_determinant();
    assert_eq(&determinant, &tensor_rank_2.determinant())?;
    assert_eq(&inverse, &tensor_rank_2.inverse())
}

#[test]
fn inverse_transpose_dim_2() -> Result<(), TestError> {
    assert_eq_within_tols(
        &(get_tensor_rank_2_dim_2().transpose() * get_tensor_rank_2_dim_2().inverse_transpose()),
        &TensorRank2::identity(),
    )
}

#[test]
fn inverse_transpose_dim_3() -> Result<(), TestError> {
    assert_eq_within_tols(
        &(get_tensor_rank_2_dim_3().transpose() * get_tensor_rank_2_dim_3().inverse_transpose()),
        &TensorRank2::identity(),
    )
}

#[test]
fn inverse_transpose_dim_4() -> Result<(), TestError> {
    assert_eq_within_tols(
        &(get_tensor_rank_2_dim_4().transpose() * get_tensor_rank_2_dim_4().inverse_transpose()),
        &TensorRank2::identity(),
    )
}

#[test]
fn inverse_transpose_9() -> Result<(), TestError> {
    assert_eq_within_tols(
        &(get_tensor_rank_2_dim_9().transpose() * get_tensor_rank_2_dim_9().inverse_transpose()),
        &TensorRank2::identity(),
    )
}

#[test]
fn inverse_transpose_and_determinant_dim_2() -> Result<(), TestError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_2();
    let (inverse_transpose, determinant) = tensor_rank_2.inverse_transpose_and_determinant();
    assert_eq(&determinant, &tensor_rank_2.determinant())?;
    assert_eq(&inverse_transpose, &tensor_rank_2.inverse_transpose())
}

#[test]
fn inverse_transpose_and_determinant_dim_3() -> Result<(), TestError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_3();
    let (inverse_transpose, determinant) = tensor_rank_2.inverse_transpose_and_determinant();
    assert_eq(&determinant, &tensor_rank_2.determinant())?;
    assert_eq(&inverse_transpose, &tensor_rank_2.inverse_transpose())
}

#[test]
fn inverse_transpose_and_determinant_dim_4() -> Result<(), TestError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_4();
    let (inverse_transpose, determinant) = tensor_rank_2.inverse_transpose_and_determinant();
    assert_eq(&determinant, &tensor_rank_2.determinant())?;
    assert_eq(&inverse_transpose, &tensor_rank_2.inverse_transpose())
}

#[test]
fn inverse_transpose_and_determinant_dim_9() -> Result<(), TestError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_9();
    let (inverse_transpose, determinant) = tensor_rank_2.inverse_transpose_and_determinant();
    assert_eq(&determinant, &tensor_rank_2.determinant())?;
    assert_eq(&inverse_transpose, &tensor_rank_2.inverse_transpose())
}

#[test]
fn lu_decomposition() -> Result<(), TestError> {
    let (l, u, p) = get_tensor_rank_2_dim_9().lu_decomposition();
    l.iter()
        .enumerate()
        .zip(u.iter())
        .for_each(|((i, l_i), u_i)| {
            l_i.iter()
                .enumerate()
                .zip(u_i.iter())
                .for_each(|((j, l_ij), u_ij)| match i.cmp(&j) {
                    Ordering::Equal => assert_eq!(l_ij, &1.0),
                    Ordering::Greater => assert_eq!(u_ij, &0.0),
                    Ordering::Less => assert_eq!(l_ij, &0.0),
                })
        });
    assert_eq_within_tols(
        &(l * u),
        &p.iter()
            .map(|&p_i| get_tensor_rank_2_dim_9()[p_i].clone())
            .collect(),
    )
}
