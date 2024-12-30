use super::{
    super::{
        test::{assert_eq, assert_eq_within_tols, ErrorTensor, TestError},
        TensorError,
    },
    Rank2, Tensor, TensorArray, TensorRank0, TensorRank1, TensorRank1List, TensorRank2,
    TensorRank2List2D, TensorRank4,
};
use crate::{ABS_TOL, REL_TOL};
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
        [0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 4.0, 2.0, 1.0],
        [3.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 4.0, 2.0],
        [4.0, 4.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 4.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, 3.0, 0.0, 1.0, 1.0],
        [4.0, 2.0, 3.0, 4.0, 2.0, 4.0, 3.0, 0.0, 4.0],
        [1.0, 3.0, 2.0, 0.0, 0.0, 0.0, 2.0, 4.0, 2.0],
        [2.0, 2.0, 2.0, 4.0, 1.0, 2.0, 4.0, 2.0, 2.0],
        [1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 4.0, 2.0, 1.0],
    ]
}

fn get_tensor_rank_1_a() -> TensorRank1<4, 1> {
    TensorRank1::new([1.0, 2.0, 3.0, 4.0])
}

fn get_tensor_rank_1_b() -> TensorRank1<4, 1> {
    TensorRank1::new([5.0, 7.0, 6.0, 8.0])
}

fn get_tensor_rank_2_dim_2() -> TensorRank2<2, 1, 1> {
    TensorRank2::new(get_array_dim_2())
}

fn get_tensor_rank_2_dim_3() -> TensorRank2<3, 1, 1> {
    TensorRank2::new(get_array_dim_3())
}

fn get_tensor_rank_2_pos_def_dim_3() -> TensorRank2<3, 1, 1> {
    TensorRank2::new([
        [1.25408849, 0.87253221, 1.02098481],
        [0.87253221, 0.73474241, 0.82412859],
        [1.02098481, 0.82412859, 1.22438759],
    ])
}

fn get_tensor_rank_2_pos_def_dim_4() -> TensorRank2<4, 1, 1> {
    TensorRank2::new([
        [1.45127677, 0.8346131, 1.0384431, 1.01387535],
        [0.8346131, 1.05044889, 0.98065684, 1.10580611],
        [1.0384431, 0.98065684, 1.167509, 0.98059065],
        [1.01387535, 1.10580611, 0.98059065, 1.33644059],
    ])
}

fn get_tensor_rank_2_pos_def_dim_3_cholesky_decomposition() -> TensorRank2<3, 1, 1> {
    TensorRank2::new([
        [1.1198609244008828, 0.0, 0.0],
        [0.7791433659200121, 0.3573206198120032, 0.0],
        [0.9117067912216147, 0.3184207281833315, 0.5401727100692741],
    ])
}

fn get_tensor_rank_2_pos_def_dim_4_cholesky_decomposition() -> TensorRank2<4, 1, 1> {
    TensorRank2::new([
        [1.204689491113789, 0.0, 0.0, 0.0],
        [0.6928035034391833, 0.7552960979790598, 0.0, 0.0],
        [
            0.8620006297555672,
            0.5076946442124899,
            0.40830143587667994,
            0.0,
        ],
        [
            0.8416072004269142,
            0.6920963770675851,
            -0.23573003235768966,
            0.3058951893772402,
        ],
    ])
}

fn get_tensor_rank_2_dim_4() -> TensorRank2<4, 1, 1> {
    TensorRank2::new(get_array_dim_4())
}

fn get_tensor_rank_2_dim_9() -> TensorRank2<9, 1, 1> {
    TensorRank2::new(get_array_dim_9())
}

fn get_other_tensor_rank_2_dim_2() -> TensorRank2<2, 1, 1> {
    TensorRank2::new([[5.0, 6.0], [7.0, 8.0]])
}

fn get_other_tensor_rank_2_dim_3() -> TensorRank2<3, 1, 1> {
    TensorRank2::new([[3.0, 2.0, 3.0], [6.0, 5.0, 2.0], [4.0, 5.0, 0.0]])
}

fn get_other_tensor_rank_2_dim_4() -> TensorRank2<4, 1, 1> {
    TensorRank2::new([
        [3.0, 2.0, 3.0, 5.0],
        [6.0, 5.0, 2.0, 4.0],
        [4.0, 5.0, 0.0, 4.0],
        [4.0, 4.0, 1.0, 6.0],
    ])
}

fn get_diagonal_tensor_rank_2_dim_4() -> TensorRank2<4, 1, 1> {
    TensorRank2::new([
        [3.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 6.0],
    ])
}

fn get_other_tensor_rank_2_dim_9() -> TensorRank2<9, 1, 1> {
    TensorRank2::new([
        [0.0, 4.0, 2.0, 0.0, 1.0, 4.0, 2.0, 4.0, 1.0],
        [1.0, 2.0, 2.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0],
        [3.0, 0.0, 2.0, 3.0, 3.0, 0.0, 0.0, 0.0, 2.0],
        [2.0, 3.0, 0.0, 0.0, 1.0, 3.0, 3.0, 4.0, 2.0],
        [0.0, 4.0, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0, 1.0],
        [1.0, 3.0, 0.0, 3.0, 3.0, 2.0, 1.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 1.0, 3.0, 4.0],
        [2.0, 0.0, 4.0, 3.0, 1.0, 2.0, 0.0, 3.0, 4.0],
        [4.0, 2.0, 0.0, 0.0, 4.0, 0.0, 4.0, 2.0, 2.0],
    ])
}

fn get_other_tensor_rank_2_mul_tensor_rank_1_dim_4() -> TensorRank1<4, 1> {
    TensorRank1::new([51.0, 14.0, 22.0, 27.0])
}

fn get_other_tensor_rank_2_add_tensor_rank_2_dim_4() -> TensorRank2<4, 1, 1> {
    TensorRank2::new([
        [4.0, 6.0, 9.0, 11.0],
        [7.0, 10.0, 3.0, 4.0],
        [5.0, 8.0, 5.0, 4.0],
        [5.0, 8.0, 7.0, 6.0],
    ])
}

fn get_other_tensor_rank_2_sub_tensor_rank_2_dim_4() -> TensorRank2<4, 1, 1> {
    TensorRank2::new([
        [-2.0, 2.0, 3.0, 1.0],
        [-5.0, 0.0, -1.0, -4.0],
        [-3.0, -2.0, 5.0, -4.0],
        [-3.0, 0.0, 5.0, -6.0],
    ])
}

fn get_other_tensor_rank_2_mul_tensor_rank_2_dim_4() -> TensorRank2<4, 1, 1> {
    TensorRank2::new([
        [75.0, 76.0, 17.0, 81.0],
        [37.0, 32.0, 13.0, 29.0],
        [41.0, 42.0, 9.0, 37.0],
        [51.0, 52.0, 11.0, 45.0],
    ])
}

fn get_tensor_rank_1_list() -> TensorRank1List<3, 1, 8> {
    TensorRank1List::new([
        [5.0, 0.0, 0.0],
        [5.0, 5.0, 6.0],
        [3.0, 1.0, 4.0],
        [3.0, 4.0, 2.0],
        [1.0, 0.0, 3.0],
        [1.0, 3.0, 1.0],
        [1.0, 6.0, 0.0],
        [1.0, 1.0, 1.0],
    ])
}

fn get_tensor_rank_2_list_2d() -> TensorRank2List2D<3, 1, 1, 2, 2> {
    TensorRank2List2D::new([
        [
            [[1.0, 4.0, 6.0], [7.0, 2.0, 5.0], [9.0, 8.0, 3.0]],
            [[3.0, 2.0, 3.0], [6.0, 5.0, 2.0], [4.0, 5.0, 0.0]],
        ],
        [
            [[5.0, 2.0, 9.0], [2.0, 4.0, 5.0], [1.0, 3.0, 8.0]],
            [[4.0, 3.0, 2.0], [2.0, 5.0, 4.0], [1.0, 7.0, 1.0]],
        ],
    ])
}

fn get_tensor_rank_2_mul_tensor_rank_2_list_2d() -> TensorRank2List2D<3, 1, 1, 2, 2> {
    TensorRank2List2D::new([
        [
            [[83.0, 60.0, 44.0], [66.0, 72.0, 67.0], [92.0, 76.0, 103.0]],
            [[51.0, 52.0, 11.0], [53.0, 49.0, 25.0], [87.0, 73.0, 43.0]],
        ],
        [
            [[19.0, 36.0, 77.0], [44.0, 37.0, 113.0], [64.0, 59.0, 145.0]],
            [[18.0, 65.0, 24.0], [37.0, 66.0, 27.0], [55.0, 88.0, 53.0]],
        ],
    ])
}

fn get_tensor_rank_4() -> TensorRank4<3, 1, 1, 2, 3> {
    TensorRank4::new([
        [
            [[7.0, 3.0, 7.0], [3.0, 2.0, 7.0], [9.0, 8.0, 4.0]],
            [[1.0, 10.0, 7.0], [0.0, 3.0, 3.0], [4.0, 8.0, 8.0]],
            [[0.0, 1.0, 7.0], [1.0, 2.0, 9.0], [3.0, 5.0, 4.0]],
        ],
        [
            [[2.0, 1.0, 8.0], [6.0, 2.0, 6.0], [4.0, 6.0, 2.0]],
            [[7.0, 7.0, 8.0], [8.0, 4.0, 4.0], [10.0, 9.0, 9.0]],
            [[3.0, 3.0, 3.0], [1.0, 4.0, 3.0], [10.0, 9.0, 5.0]],
        ],
        [
            [[9.0, 5.0, 1.0], [7.0, 9.0, 9.0], [5.0, 9.0, 10.0]],
            [[5.0, 9.0, 0.0], [4.0, 5.0, 7.0], [5.0, 4.0, 7.0]],
            [[1.0, 2.0, 7.0], [8.0, 2.0, 6.0], [2.0, 7.0, 5.0]],
        ],
    ])
}

fn get_tensor_rank_2_div_tensor_rank_4() -> TensorRank2<3, 2, 3> {
    TensorRank2::new([
        [-0.8591023283605275, 0.5463144610682097, 0.48148464803521684],
        [0.14461826142457423, 2.8819091589827597, 0.3555608669979796],
        [
            0.29609312727618836,
            -0.4778620587076813,
            -1.3810401169942013,
        ],
    ])
}

#[test]
fn add_tensor_rank_2_to_self() -> Result<(), TestError> {
    assert_eq(
        &(get_tensor_rank_2_dim_4() + get_other_tensor_rank_2_dim_4()),
        &get_other_tensor_rank_2_add_tensor_rank_2_dim_4(),
    )
}

#[test]
fn add_tensor_rank_2_ref_to_self() -> Result<(), TestError> {
    assert_eq(
        &(get_tensor_rank_2_dim_4() + &get_other_tensor_rank_2_dim_4()),
        &get_other_tensor_rank_2_add_tensor_rank_2_dim_4(),
    )
}

#[test]
fn add_tensor_rank_2_to_self_ref() -> Result<(), TestError> {
    assert_eq(
        &(&get_tensor_rank_2_dim_4() + get_other_tensor_rank_2_dim_4()),
        &get_other_tensor_rank_2_add_tensor_rank_2_dim_4(),
    )
}

#[test]
fn add_assign_tensor_rank_2() -> Result<(), TestError> {
    let mut tensor_rank_2 = get_tensor_rank_2_dim_4();
    tensor_rank_2 += get_other_tensor_rank_2_dim_4();
    assert_eq(
        &tensor_rank_2,
        &get_other_tensor_rank_2_add_tensor_rank_2_dim_4(),
    )
}

#[test]
fn add_assign_tensor_rank_2_ref() -> Result<(), TestError> {
    let mut tensor_rank_2 = get_tensor_rank_2_dim_4();
    tensor_rank_2 += &get_other_tensor_rank_2_dim_4();
    assert_eq(
        &tensor_rank_2,
        &get_other_tensor_rank_2_add_tensor_rank_2_dim_4(),
    )
}

#[test]
fn as_array_dim_2() {
    assert_eq!(get_tensor_rank_2_dim_2().as_array(), get_array_dim_2())
}

#[test]
fn as_array_dim_3() {
    assert_eq!(get_tensor_rank_2_dim_3().as_array(), get_array_dim_3())
}

#[test]
fn as_array_dim_4() {
    assert_eq!(get_tensor_rank_2_dim_4().as_array(), get_array_dim_4())
}

#[test]
fn as_array_dim_9() {
    assert_eq!(get_tensor_rank_2_dim_9().as_array(), get_array_dim_9())
}

#[test]
fn cholesky_decomposition_fail_dim_2() {
    assert_eq!(
        get_tensor_rank_2_dim_2().cholesky_decomposition(),
        Err(TensorError::NotPositiveDefinite),
    )
}

#[test]
fn cholesky_decomposition_dim_3() -> Result<(), TestError> {
    let tensor = get_tensor_rank_2_pos_def_dim_3();
    let lower_diagonal = tensor.cholesky_decomposition()?;
    assert_eq(
        &lower_diagonal,
        &get_tensor_rank_2_pos_def_dim_3_cholesky_decomposition(),
    )?;
    assert_eq_within_tols(&tensor, &(&lower_diagonal * lower_diagonal.transpose()))
}

#[test]
fn cholesky_decomposition_fail_dim_3() {
    assert_eq!(
        get_tensor_rank_2_dim_3().cholesky_decomposition(),
        Err(TensorError::NotPositiveDefinite),
    )
}

#[test]
fn cholesky_decomposition_dim_4() -> Result<(), TestError> {
    let tensor = get_tensor_rank_2_pos_def_dim_4();
    let lower_diagonal = tensor.cholesky_decomposition()?;
    assert_eq(
        &lower_diagonal,
        &get_tensor_rank_2_pos_def_dim_4_cholesky_decomposition(),
    )?;
    assert_eq_within_tols(&tensor, &(&lower_diagonal * lower_diagonal.transpose()))
}

#[test]
fn cholesky_decomposition_fail_dim_4() {
    assert_eq!(
        get_tensor_rank_2_dim_4().cholesky_decomposition(),
        Err(TensorError::NotPositiveDefinite),
    )
}

#[test]
fn cholesky_decomposition_fail_dim_9() {
    assert_eq!(
        get_tensor_rank_2_dim_9().cholesky_decomposition(),
        Err(TensorError::NotPositiveDefinite),
    )
}

#[test]
fn div_tensor_rank_4_to_self() -> Result<(), TestError> {
    assert_eq(
        &(get_tensor_rank_2_dim_3() / get_tensor_rank_4()),
        &get_tensor_rank_2_div_tensor_rank_4(),
    )
}

#[test]
fn div_tensor_rank_0_to_self() -> Result<(), TestError> {
    (get_tensor_rank_2_dim_4() / 3.3)
        .iter()
        .zip(get_array_dim_4().iter())
        .try_for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i.iter().zip(array_i.iter()).try_for_each(
                |(tensor_rank_2_ij, array_ij)| assert_eq(tensor_rank_2_ij, &(array_ij / 3.3)),
            )
        })?;
    Ok(())
}

#[test]
fn div_tensor_rank_0_to_self_ref() -> Result<(), TestError> {
    (&get_tensor_rank_2_dim_4() / 3.3)
        .iter()
        .zip(get_array_dim_4().iter())
        .try_for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i.iter().zip(array_i.iter()).try_for_each(
                |(tensor_rank_2_ij, array_ij)| assert_eq(tensor_rank_2_ij, &(array_ij / 3.3)),
            )
        })?;
    Ok(())
}

#[test]
#[allow(clippy::op_ref)]
fn div_tensor_rank_0_ref_to_self() -> Result<(), TestError> {
    (get_tensor_rank_2_dim_4() / &3.3)
        .iter()
        .zip(get_array_dim_4().iter())
        .try_for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i.iter().zip(array_i.iter()).try_for_each(
                |(tensor_rank_2_ij, array_ij)| assert_eq(tensor_rank_2_ij, &(array_ij / 3.3)),
            )
        })?;
    Ok(())
}

#[test]
#[allow(clippy::op_ref)]
fn div_tensor_rank_0_ref_to_self_ref() -> Result<(), TestError> {
    (&get_tensor_rank_2_dim_4() / &3.3)
        .iter()
        .zip(get_array_dim_4().iter())
        .try_for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i.iter().zip(array_i.iter()).try_for_each(
                |(tensor_rank_2_ij, array_ij)| assert_eq(tensor_rank_2_ij, &(array_ij / 3.3)),
            )
        })?;
    Ok(())
}

#[test]
fn div_assign_tensor_rank_0() -> Result<(), TestError> {
    let mut tensor_rank_2 = get_tensor_rank_2_dim_4();
    tensor_rank_2 /= 3.3;
    tensor_rank_2
        .iter()
        .zip(get_array_dim_4().iter())
        .try_for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i.iter().zip(array_i.iter()).try_for_each(
                |(tensor_rank_2_ij, array_ij)| assert_eq(tensor_rank_2_ij, &(array_ij / 3.3)),
            )
        })?;
    Ok(())
}

#[test]
fn div_assign_tensor_rank_0_ref() -> Result<(), TestError> {
    let mut tensor_rank_2 = get_tensor_rank_2_dim_4();
    tensor_rank_2 /= &3.3;
    tensor_rank_2
        .iter()
        .zip(get_array_dim_4().iter())
        .try_for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i.iter().zip(array_i.iter()).try_for_each(
                |(tensor_rank_2_ij, array_ij)| assert_eq(tensor_rank_2_ij, &(array_ij / 3.3)),
            )
        })?;
    Ok(())
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
    assert_eq_within_tols(&get_tensor_rank_2_dim_9().determinant(), &2398.0)
}

#[test]
fn deviatoric_dim_2() -> Result<(), TestError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_2();
    let trace = tensor_rank_2.trace();
    let deviatoric_tensor_rank_2 = tensor_rank_2.deviatoric();
    assert_eq(&deviatoric_tensor_rank_2.trace(), &0.0)?;
    assert_eq(
        &deviatoric_tensor_rank_2,
        &(tensor_rank_2 - TensorRank2::identity() * (trace / 2.0)),
    )
}

#[test]
fn deviatoric_dim_3() -> Result<(), TestError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_3();
    let trace = tensor_rank_2.trace();
    let deviatoric_tensor_rank_2 = tensor_rank_2.deviatoric();
    assert_eq(&deviatoric_tensor_rank_2.trace(), &0.0)?;
    assert_eq(
        &deviatoric_tensor_rank_2,
        &(tensor_rank_2 - TensorRank2::identity() * (trace / 3.0)),
    )
}

#[test]
fn deviatoric_dim_4() -> Result<(), TestError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_4();
    let trace = tensor_rank_2.trace();
    let deviatoric_tensor_rank_2 = tensor_rank_2.deviatoric();
    assert_eq(&deviatoric_tensor_rank_2.trace(), &0.0)?;
    assert_eq(
        &deviatoric_tensor_rank_2,
        &(tensor_rank_2 - TensorRank2::identity() * (trace / 4.0)),
    )
}

#[test]
fn deviatoric_dim_9() -> Result<(), TestError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_9();
    let trace = tensor_rank_2.trace();
    let deviatoric_tensor_rank_2 = tensor_rank_2.deviatoric();
    assert_eq(&deviatoric_tensor_rank_2.trace(), &0.0)?;
    assert_eq(
        &deviatoric_tensor_rank_2,
        &(tensor_rank_2 - TensorRank2::identity() * (trace / 9.0)),
    )
}

#[test]
fn deviatoric_and_trace_dim_2() -> Result<(), TestError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_2();
    let (deviatoric, trace) = tensor_rank_2.deviatoric_and_trace();
    assert_eq(&tensor_rank_2.trace(), &trace)?;
    assert_eq(&tensor_rank_2.deviatoric(), &deviatoric)
}

#[test]
fn deviatoric_and_trace_dim_3() -> Result<(), TestError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_3();
    let (deviatoric, trace) = tensor_rank_2.deviatoric_and_trace();
    assert_eq(&tensor_rank_2.trace(), &trace)?;
    assert_eq(&tensor_rank_2.deviatoric(), &deviatoric)
}

#[test]
fn deviatoric_and_trace_dim_4() -> Result<(), TestError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_4();
    let (deviatoric, trace) = tensor_rank_2.deviatoric_and_trace();
    assert_eq(&tensor_rank_2.trace(), &trace)?;
    assert_eq(&tensor_rank_2.deviatoric(), &deviatoric)
}

#[test]
fn deviatoric_and_trace_dim_9() -> Result<(), TestError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_9();
    let (deviatoric, trace) = tensor_rank_2.deviatoric_and_trace();
    assert_eq(&tensor_rank_2.trace(), &trace)?;
    assert_eq(&tensor_rank_2.deviatoric(), &deviatoric)
}

#[test]
fn dyad() {
    let tensor_rank_1_a = get_tensor_rank_1_a();
    let tensor_rank_1_b = get_tensor_rank_1_b();
    let tensor_rank_2 = TensorRank2::dyad(&tensor_rank_1_a, &tensor_rank_1_b);
    tensor_rank_2.iter().zip(tensor_rank_1_a.iter()).for_each(
        |(tensor_rank_2_i, tensor_rank_1_a_i)| {
            tensor_rank_2_i.iter().zip(tensor_rank_1_b.iter()).for_each(
                |(tensor_rank_2_ij, tensor_rank_1_b_j)| {
                    assert_eq!(tensor_rank_2_ij, &(tensor_rank_1_a_i * tensor_rank_1_b_j))
                },
            )
        },
    );
}

#[test]
fn error() {
    let a = get_tensor_rank_1_a();
    let b = get_tensor_rank_1_b();
    assert_eq!(a.error(&a, &ABS_TOL, &REL_TOL), None);
    assert_eq!(a.error(&b, &ABS_TOL, &REL_TOL), Some(4));
}

#[test]
fn from_iter() {
    let into_iterator = get_tensor_rank_2_dim_4().0.into_iter();
    let tensor_rank_2 = TensorRank2::<4, 1, 1>::from_iter(get_tensor_rank_2_dim_4().0);
    tensor_rank_2
        .iter()
        .zip(into_iterator)
        .for_each(|(tensor_rank_2_i, value_i)| {
            tensor_rank_2_i
                .iter()
                .zip(value_i.iter())
                .for_each(|(tensor_rank_2_ij, value_ij)| assert_eq!(tensor_rank_2_ij, value_ij))
        });
}

#[test]
fn full_contraction_dim_2() -> Result<(), TestError> {
    assert_eq_within_tols(
        &get_tensor_rank_2_dim_2().full_contraction(&get_other_tensor_rank_2_dim_2()),
        &70.0,
    )
}

#[test]
fn full_contraction_dim_3() -> Result<(), TestError> {
    assert_eq_within_tols(
        &get_tensor_rank_2_dim_3().full_contraction(&get_other_tensor_rank_2_dim_3()),
        &167.0,
    )
}

#[test]
fn full_contraction_dim_4() -> Result<(), TestError> {
    assert_eq_within_tols(
        &get_tensor_rank_2_dim_4().full_contraction(&get_other_tensor_rank_2_dim_4()),
        &137.0,
    )
}

#[test]
fn full_contraction_dim_9() -> Result<(), TestError> {
    assert_eq_within_tols(
        &get_tensor_rank_2_dim_9().full_contraction(&get_other_tensor_rank_2_dim_9()),
        &269.0,
    )
}

#[test]
fn identity() {
    TensorRank2::<9, 1, 1>::identity()
        .iter()
        .enumerate()
        .for_each(|(i, tensor_rank_2_i)| {
            tensor_rank_2_i
                .iter()
                .enumerate()
                .for_each(|(j, tensor_rank_2_ij)| {
                    if i == j {
                        assert_eq!(tensor_rank_2_ij, &1.0)
                    } else {
                        assert_eq!(tensor_rank_2_ij, &0.0)
                    }
                })
        });
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
fn is_diagonal() {
    assert!(get_diagonal_tensor_rank_2_dim_4().is_diagonal())
}

#[test]
fn is_not_diagonal() {
    assert!(!get_other_tensor_rank_2_dim_4().is_diagonal())
}

#[test]
fn is_diagonal_identity() {
    assert!(TensorRank2::<3, 0, 0>::identity().is_diagonal())
}

#[test]
fn is_diagonal_zero() {
    assert!(TensorRank2::<4, 1, 1>::zero().is_diagonal())
}

#[test]
fn is_identity_dim_3() {
    assert!(TensorRank2::<3, 0, 0>::identity().is_identity())
}

#[test]
fn is_not_identity_dim_3() {
    assert!(!TensorRank2::<3, 0, 0>::zero().is_identity())
}

#[test]
fn is_identity_dim_4() {
    assert!(TensorRank2::<4, 1, 1>::identity().is_identity())
}

#[test]
fn is_not_identity_dim_4() {
    assert!(!get_diagonal_tensor_rank_2_dim_4().is_identity())
}

#[test]
fn is_zero_dim_3() {
    assert!(TensorRank2::<3, 0, 0>::zero().is_zero())
}

#[test]
fn is_not_zero_dim_3() {
    assert!(!TensorRank2::<3, 0, 0>::identity().is_zero())
}

#[test]
fn is_zero_dim_4() {
    assert!(TensorRank2::<4, 1, 1>::zero().is_zero())
}

#[test]
fn is_not_zero_dim_4() {
    assert!(!get_other_tensor_rank_2_dim_4().is_zero())
}

#[test]
fn iter() {
    get_tensor_rank_2_dim_4()
        .iter()
        .zip(get_array_dim_4().iter())
        .for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i
                .iter()
                .zip(array_i.iter())
                .for_each(|(tensor_rank_2_ij, array_ij)| assert_eq!(tensor_rank_2_ij, array_ij))
        });
}

#[test]
fn iter_mut() {
    get_tensor_rank_2_dim_4()
        .iter_mut()
        .zip(get_array_dim_4().iter_mut())
        .for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i
                .iter_mut()
                .zip(array_i.iter_mut())
                .for_each(|(tensor_rank_2_ij, array_ij)| assert_eq!(tensor_rank_2_ij, array_ij))
        });
}

#[test]
fn lu_decomposition() {
    let (tensor_l, tensor_u) = get_tensor_rank_2_dim_9().lu_decomposition();
    tensor_l
        .iter()
        .enumerate()
        .zip(tensor_u.iter())
        .for_each(|((i, tensor_l_i), tensor_u_i)| {
            tensor_l_i
                .iter()
                .enumerate()
                .zip(tensor_u_i.iter())
                .for_each(|((j, tensor_l_ij), tensor_u_ij)| match i.cmp(&j) {
                    Ordering::Greater => assert_eq!(tensor_u_ij, &0.0),
                    Ordering::Less => assert_eq!(tensor_l_ij, &0.0),
                    _ => (),
                })
        });
}

#[test]
fn mul_tensor_rank_0_to_self() {
    (get_tensor_rank_2_dim_4() * 3.3)
        .iter()
        .zip(get_array_dim_4().iter())
        .for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i
                .iter()
                .zip(array_i.iter())
                .for_each(|(tensor_rank_2_ij, array_ij)| {
                    assert_eq!(tensor_rank_2_ij, &(array_ij * 3.3))
                })
        });
}

#[test]
fn mul_tensor_rank_0_to_self_ref() {
    (&get_tensor_rank_2_dim_4() * 3.3)
        .iter()
        .zip(get_array_dim_4().iter())
        .for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i
                .iter()
                .zip(array_i.iter())
                .for_each(|(tensor_rank_2_ij, array_ij)| {
                    assert_eq!(tensor_rank_2_ij, &(array_ij * 3.3))
                })
        });
}

#[test]
#[allow(clippy::op_ref)]
fn mul_tensor_rank_0_ref_to_self() {
    (get_tensor_rank_2_dim_4() * &3.3)
        .iter()
        .zip(get_array_dim_4().iter())
        .for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i
                .iter()
                .zip(array_i.iter())
                .for_each(|(tensor_rank_2_ij, array_ij)| {
                    assert_eq!(tensor_rank_2_ij, &(array_ij * 3.3))
                })
        });
}

#[test]
#[allow(clippy::op_ref)]
fn mul_tensor_rank_0_ref_to_self_ref() {
    (&get_tensor_rank_2_dim_4() * &3.3)
        .iter()
        .zip(get_array_dim_4().iter())
        .for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i
                .iter()
                .zip(array_i.iter())
                .for_each(|(tensor_rank_2_ij, array_ij)| {
                    assert_eq!(tensor_rank_2_ij, &(array_ij * 3.3))
                })
        });
}

#[test]
fn mul_assign_tensor_rank_0() {
    let mut tensor_rank_2 = get_tensor_rank_2_dim_4();
    tensor_rank_2 *= 3.3;
    tensor_rank_2
        .iter()
        .zip(get_array_dim_4().iter())
        .for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i
                .iter()
                .zip(array_i.iter())
                .for_each(|(tensor_rank_2_ij, array_ij)| {
                    assert_eq!(tensor_rank_2_ij, &(array_ij * 3.3))
                })
        });
}

#[test]
fn mul_assign_tensor_rank_0_ref() {
    let mut tensor_rank_2 = get_tensor_rank_2_dim_4();
    tensor_rank_2 *= &3.3;
    tensor_rank_2
        .iter()
        .zip(get_array_dim_4().iter())
        .for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i
                .iter()
                .zip(array_i.iter())
                .for_each(|(tensor_rank_2_ij, array_ij)| {
                    assert_eq!(tensor_rank_2_ij, &(array_ij * 3.3))
                })
        });
}

#[test]
fn mul_tensor_rank_1_to_self() {
    (get_tensor_rank_2_dim_4() * get_tensor_rank_1_a())
        .iter()
        .zip(get_other_tensor_rank_2_mul_tensor_rank_1_dim_4().iter())
        .for_each(|(tensor_rank_1_i, res_tensor_rank_1_i)| {
            assert_eq!(tensor_rank_1_i, res_tensor_rank_1_i)
        });
}

#[test]
fn mul_tensor_rank_1_ref_to_self() {
    (get_tensor_rank_2_dim_4() * &get_tensor_rank_1_a())
        .iter()
        .zip(get_other_tensor_rank_2_mul_tensor_rank_1_dim_4().iter())
        .for_each(|(tensor_rank_1_i, res_tensor_rank_1_i)| {
            assert_eq!(tensor_rank_1_i, res_tensor_rank_1_i)
        });
}

#[test]
fn mul_tensor_rank_1_to_self_ref() {
    (&get_tensor_rank_2_dim_4() * get_tensor_rank_1_a())
        .iter()
        .zip(get_other_tensor_rank_2_mul_tensor_rank_1_dim_4().iter())
        .for_each(|(tensor_rank_1_i, res_tensor_rank_1_i)| {
            assert_eq!(tensor_rank_1_i, res_tensor_rank_1_i)
        });
}

#[test]
fn mul_tensor_rank_1_ref_to_self_ref() {
    (&get_tensor_rank_2_dim_4() * &get_tensor_rank_1_a())
        .iter()
        .zip(get_other_tensor_rank_2_mul_tensor_rank_1_dim_4().iter())
        .for_each(|(tensor_rank_1_i, res_tensor_rank_1_i)| {
            assert_eq!(tensor_rank_1_i, res_tensor_rank_1_i)
        });
}

#[test]
fn mul_tensor_rank_2_to_self() {
    (get_tensor_rank_2_dim_4() * get_other_tensor_rank_2_dim_4())
        .iter()
        .zip(get_other_tensor_rank_2_mul_tensor_rank_2_dim_4().iter())
        .for_each(|(tensor_rank_2_i, res_tensor_rank_2_i)| {
            tensor_rank_2_i
                .iter()
                .zip(res_tensor_rank_2_i.iter())
                .for_each(|(tensor_rank_2_ij, res_tensor_rank_2_ij)| {
                    assert_eq!(tensor_rank_2_ij, res_tensor_rank_2_ij)
                })
        });
}

#[test]
fn mul_tensor_rank_2_ref_to_self() {
    (get_tensor_rank_2_dim_4() * &get_other_tensor_rank_2_dim_4())
        .iter()
        .zip(get_other_tensor_rank_2_mul_tensor_rank_2_dim_4().iter())
        .for_each(|(tensor_rank_2_i, res_tensor_rank_2_i)| {
            tensor_rank_2_i
                .iter()
                .zip(res_tensor_rank_2_i.iter())
                .for_each(|(tensor_rank_2_ij, res_tensor_rank_2_ij)| {
                    assert_eq!(tensor_rank_2_ij, res_tensor_rank_2_ij)
                })
        });
}

#[test]
fn mul_tensor_rank_2_to_self_ref() {
    (&get_tensor_rank_2_dim_4() * get_other_tensor_rank_2_dim_4())
        .iter()
        .zip(get_other_tensor_rank_2_mul_tensor_rank_2_dim_4().iter())
        .for_each(|(tensor_rank_2_i, res_tensor_rank_2_i)| {
            tensor_rank_2_i
                .iter()
                .zip(res_tensor_rank_2_i.iter())
                .for_each(|(tensor_rank_2_ij, res_tensor_rank_2_ij)| {
                    assert_eq!(tensor_rank_2_ij, res_tensor_rank_2_ij)
                })
        });
}

#[test]
fn mul_tensor_rank_2_ref_to_self_ref() {
    (&get_tensor_rank_2_dim_4() * &get_other_tensor_rank_2_dim_4())
        .iter()
        .zip(get_other_tensor_rank_2_mul_tensor_rank_2_dim_4().iter())
        .for_each(|(tensor_rank_2_i, res_tensor_rank_2_i)| {
            tensor_rank_2_i
                .iter()
                .zip(res_tensor_rank_2_i.iter())
                .for_each(|(tensor_rank_2_ij, res_tensor_rank_2_ij)| {
                    assert_eq!(tensor_rank_2_ij, res_tensor_rank_2_ij)
                })
        });
}

#[test]
fn mul_tensor_rank_1_list_to_self() {
    (get_tensor_rank_2_dim_3() * get_tensor_rank_1_list())
        .iter()
        .zip(get_tensor_rank_1_list().iter())
        .for_each(|(res_tensor_rank_1, tensor_rank_1)| {
            res_tensor_rank_1
                .iter()
                .zip((get_tensor_rank_2_dim_3() * tensor_rank_1).iter())
                .for_each(|(res_tensor_rank_1_i, tensor_rank_1_i)| {
                    assert_eq!(res_tensor_rank_1_i, tensor_rank_1_i)
                })
        })
}

#[test]
fn mul_tensor_rank_1_list_ref_to_self() {
    (get_tensor_rank_2_dim_3() * &get_tensor_rank_1_list())
        .iter()
        .zip(get_tensor_rank_1_list().iter())
        .for_each(|(res_tensor_rank_1, tensor_rank_1)| {
            res_tensor_rank_1
                .iter()
                .zip((get_tensor_rank_2_dim_3() * tensor_rank_1).iter())
                .for_each(|(res_tensor_rank_1_i, tensor_rank_1_i)| {
                    assert_eq!(res_tensor_rank_1_i, tensor_rank_1_i)
                })
        })
}

#[test]
fn mul_tensor_rank_1_list_to_self_ref() {
    (&get_tensor_rank_2_dim_3() * get_tensor_rank_1_list())
        .iter()
        .zip(get_tensor_rank_1_list().iter())
        .for_each(|(res_tensor_rank_1, tensor_rank_1)| {
            res_tensor_rank_1
                .iter()
                .zip((get_tensor_rank_2_dim_3() * tensor_rank_1).iter())
                .for_each(|(res_tensor_rank_1_i, tensor_rank_1_i)| {
                    assert_eq!(res_tensor_rank_1_i, tensor_rank_1_i)
                })
        })
}

#[test]
fn mul_tensor_rank_1_list_ref_to_self_ref() {
    (&get_tensor_rank_2_dim_3() * &get_tensor_rank_1_list())
        .iter()
        .zip(get_tensor_rank_1_list().iter())
        .for_each(|(res_tensor_rank_1, tensor_rank_1)| {
            res_tensor_rank_1
                .iter()
                .zip((get_tensor_rank_2_dim_3() * tensor_rank_1).iter())
                .for_each(|(res_tensor_rank_1_i, tensor_rank_1_i)| {
                    assert_eq!(res_tensor_rank_1_i, tensor_rank_1_i)
                })
        })
}

#[test]
fn mul_tensor_rank_2_list_2d_to_self() {
    (get_tensor_rank_2_dim_3() * get_tensor_rank_2_list_2d())
        .iter()
        .zip(get_tensor_rank_2_mul_tensor_rank_2_list_2d().iter())
        .for_each(|(tensor_rank_2_list_2d_entry, res_entry)| {
            tensor_rank_2_list_2d_entry
                .iter()
                .zip(res_entry.iter())
                .for_each(|(tensor_rank_2, res)| {
                    tensor_rank_2
                        .iter()
                        .zip(res.iter())
                        .for_each(|(tensor_rank_2_i, res_i)| {
                            tensor_rank_2_i.iter().zip(res_i.iter()).for_each(
                                |(tensor_rank_2_ij, res_ij)| assert_eq!(tensor_rank_2_ij, res_ij),
                            )
                        })
                })
        });
}

#[test]
fn mul_tensor_rank_2_list_2d_to_self_ref() {
    (&get_tensor_rank_2_dim_3() * get_tensor_rank_2_list_2d())
        .iter()
        .zip(get_tensor_rank_2_mul_tensor_rank_2_list_2d().iter())
        .for_each(|(tensor_rank_2_list_2d_entry, res_entry)| {
            tensor_rank_2_list_2d_entry
                .iter()
                .zip(res_entry.iter())
                .for_each(|(tensor_rank_2, res)| {
                    tensor_rank_2
                        .iter()
                        .zip(res.iter())
                        .for_each(|(tensor_rank_2_i, res_i)| {
                            tensor_rank_2_i.iter().zip(res_i.iter()).for_each(
                                |(tensor_rank_2_ij, res_ij)| assert_eq!(tensor_rank_2_ij, res_ij),
                            )
                        })
                })
        });
}

#[test]
fn new() {
    get_tensor_rank_2_dim_4()
        .iter()
        .zip(get_array_dim_4().iter())
        .for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i
                .iter()
                .zip(array_i.iter())
                .for_each(|(tensor_rank_2_ij, array_ij)| assert_eq!(tensor_rank_2_ij, array_ij))
        });
}

#[test]
fn norm_dim_2() -> Result<(), TestError> {
    assert_eq(&get_tensor_rank_2_dim_2().norm(), &5.477_225_575_051_661)
}

#[test]
fn norm_dim_3() -> Result<(), TestError> {
    assert_eq(&get_tensor_rank_2_dim_3().norm(), &16.881_943_016_134_134)
}

#[test]
fn norm_dim_4() -> Result<(), TestError> {
    assert_eq(&get_tensor_rank_2_dim_4().norm(), &14.282_856_857_085_7)
}

#[test]
fn norm_dim_9() -> Result<(), TestError> {
    assert_eq(&get_tensor_rank_2_dim_9().norm(), &20.976_176_963_403_03)
}

#[test]
fn size() {
    assert_eq!(
        std::mem::size_of::<TensorRank2::<3, 1, 1>>(),
        std::mem::size_of::<[TensorRank1::<3, 1>; 3]>()
    )
}

#[test]
fn second_invariant() {
    assert_eq!(get_tensor_rank_2_dim_4().second_invariant(), 16.0);
}

#[test]
fn squared_trace_dim_2() -> Result<(), TestError> {
    assert_eq_within_tols(&get_tensor_rank_2_dim_2().squared_trace(), &29.0)
}

#[test]
fn squared_trace_dim_3() -> Result<(), TestError> {
    assert_eq_within_tols(&get_tensor_rank_2_dim_3().squared_trace(), &258.0)
}

#[test]
fn squared_trace_dim_4() -> Result<(), TestError> {
    assert_eq_within_tols(&get_tensor_rank_2_dim_4().squared_trace(), &89.0)
}

#[test]
fn squared_trace_dim_9() -> Result<(), TestError> {
    assert_eq_within_tols(&get_tensor_rank_2_dim_9().squared_trace(), &318.0)
}

#[test]
fn sub_tensor_rank_2_to_self() {
    (get_tensor_rank_2_dim_4() - get_other_tensor_rank_2_dim_4())
        .iter()
        .zip(get_other_tensor_rank_2_sub_tensor_rank_2_dim_4().iter())
        .for_each(|(tensor_rank_2_i, res_tensor_rank_2_i)| {
            tensor_rank_2_i
                .iter()
                .zip(res_tensor_rank_2_i.iter())
                .for_each(|(tensor_rank_2_ij, res_tensor_rank_2_ij)| {
                    assert_eq!(tensor_rank_2_ij, res_tensor_rank_2_ij)
                })
        });
}

#[test]
fn sub_tensor_rank_2_ref_to_self() {
    (get_tensor_rank_2_dim_4() - &get_other_tensor_rank_2_dim_4())
        .iter()
        .zip(get_other_tensor_rank_2_sub_tensor_rank_2_dim_4().iter())
        .for_each(|(tensor_rank_2_i, res_tensor_rank_2_i)| {
            tensor_rank_2_i
                .iter()
                .zip(res_tensor_rank_2_i.iter())
                .for_each(|(tensor_rank_2_ij, res_tensor_rank_2_ij)| {
                    assert_eq!(tensor_rank_2_ij, res_tensor_rank_2_ij)
                })
        });
}

#[test]
fn sub_assign_tensor_rank_2() {
    let mut tensor_rank_2 = get_tensor_rank_2_dim_4();
    tensor_rank_2 -= get_other_tensor_rank_2_dim_4();
    tensor_rank_2
        .iter()
        .zip(get_other_tensor_rank_2_sub_tensor_rank_2_dim_4().iter())
        .for_each(|(tensor_rank_2_i, res_tensor_rank_2_i)| {
            tensor_rank_2_i
                .iter()
                .zip(res_tensor_rank_2_i.iter())
                .for_each(|(tensor_rank_2_ij, res_tensor_rank_2_ij)| {
                    assert_eq!(tensor_rank_2_ij, res_tensor_rank_2_ij)
                })
        });
}

#[test]
fn sub_assign_tensor_rank_2_ref() {
    let mut tensor_rank_2 = get_tensor_rank_2_dim_4();
    tensor_rank_2 -= &get_other_tensor_rank_2_dim_4();
    tensor_rank_2
        .iter()
        .zip(get_other_tensor_rank_2_sub_tensor_rank_2_dim_4().iter())
        .for_each(|(tensor_rank_2_i, res_tensor_rank_2_i)| {
            tensor_rank_2_i
                .iter()
                .zip(res_tensor_rank_2_i.iter())
                .for_each(|(tensor_rank_2_ij, res_tensor_rank_2_ij)| {
                    assert_eq!(tensor_rank_2_ij, res_tensor_rank_2_ij)
                })
        });
}

#[test]
fn trace_dim_2() -> Result<(), TestError> {
    assert_eq(&get_tensor_rank_2_dim_2().trace(), &5.0)
}

#[test]
fn trace_dim_3() -> Result<(), TestError> {
    assert_eq(&get_tensor_rank_2_dim_3().trace(), &6.0)
}

#[test]
fn trace_dim_4() -> Result<(), TestError> {
    assert_eq(&get_tensor_rank_2_dim_4().trace(), &11.0)
}

#[test]
fn trace_dim_9() -> Result<(), TestError> {
    assert_eq(&get_tensor_rank_2_dim_9().trace(), &18.0)
}

#[test]
fn transpose() {
    let tensor_rank_2 = get_tensor_rank_2_dim_4();
    let tensor_rank_2_transpose = tensor_rank_2.transpose();
    tensor_rank_2
        .iter()
        .enumerate()
        .for_each(|(i, tensor_rank_2_i)| {
            tensor_rank_2_i
                .iter()
                .enumerate()
                .for_each(|(j, tensor_rank_2_ij)| {
                    assert_eq!(tensor_rank_2_ij, &tensor_rank_2_transpose[j][i])
                })
        });
    tensor_rank_2_transpose
        .iter()
        .enumerate()
        .for_each(|(i, tensor_rank_2_transpose_i)| {
            tensor_rank_2_transpose_i.iter().enumerate().for_each(
                |(j, tensor_rank_2_transpose_ij)| {
                    assert_eq!(tensor_rank_2_transpose_ij, &tensor_rank_2[j][i])
                },
            )
        });
}

#[test]
fn zero_dim_2() {
    TensorRank2::<2, 1, 1>::zero()
        .iter()
        .for_each(|tensor_rank_2_i| {
            tensor_rank_2_i
                .iter()
                .for_each(|tensor_rank_2_ij| assert_eq!(tensor_rank_2_ij, &0.0))
        });
}

#[test]
fn zero_dim_3() {
    TensorRank2::<3, 1, 1>::zero()
        .iter()
        .for_each(|tensor_rank_2_i| {
            tensor_rank_2_i
                .iter()
                .for_each(|tensor_rank_2_ij| assert_eq!(tensor_rank_2_ij, &0.0))
        });
}

#[test]
fn zero_dim_4() {
    TensorRank2::<4, 1, 1>::zero()
        .iter()
        .for_each(|tensor_rank_2_i| {
            tensor_rank_2_i
                .iter()
                .for_each(|tensor_rank_2_ij| assert_eq!(tensor_rank_2_ij, &0.0))
        });
}

#[test]
fn zero_dim_9() {
    TensorRank2::<9, 1, 1>::zero()
        .iter()
        .for_each(|tensor_rank_2_i| {
            tensor_rank_2_i
                .iter()
                .for_each(|tensor_rank_2_ij| assert_eq!(tensor_rank_2_ij, &0.0))
        });
}
