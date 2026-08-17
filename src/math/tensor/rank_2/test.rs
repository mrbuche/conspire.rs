use super::{
    Rank2, Tensor, TensorArray, TensorRank0, TensorRank1, TensorRank1List, TensorRank2,
    TensorRank2List2D, TensorRank4,
};
use crate::math::assert::Assert;
use crate::math::{Auxiliary, Current, Intermediate, Reference};
use crate::{ABS_TOL, REL_TOL, math::assert::AssertionError};

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

fn get_tensor_rank_1_a() -> TensorRank1<4, Current> {
    TensorRank1::from([1.0, 2.0, 3.0, 4.0])
}

fn get_tensor_rank_1_b() -> TensorRank1<4, Current> {
    TensorRank1::from([5.0, 7.0, 6.0, 8.0])
}

fn get_tensor_rank_2_dim_2() -> TensorRank2<2, Current, Current> {
    TensorRank2::from(get_array_dim_2())
}

fn get_tensor_rank_2<I, J>() -> TensorRank2<3, I, J> {
    TensorRank2::from(get_array_dim_3())
}

fn get_tensor_rank_2_dim_3() -> TensorRank2<3, Current, Current> {
    TensorRank2::from(get_array_dim_3())
}

fn get_tensor_rank_2_dim_4() -> TensorRank2<4, Current, Current> {
    TensorRank2::from(get_array_dim_4())
}

fn get_tensor_rank_2_dim_9() -> TensorRank2<9, Current, Current> {
    TensorRank2::from(get_array_dim_9())
}

fn get_other_tensor_rank_2_dim_2() -> TensorRank2<2, Current, Current> {
    TensorRank2::from([[5.0, 6.0], [7.0, 8.0]])
}

fn get_other_tensor_rank_2_dim_3() -> TensorRank2<3, Current, Current> {
    TensorRank2::from([[3.0, 2.0, 3.0], [6.0, 5.0, 2.0], [4.0, 5.0, 0.0]])
}

fn get_other_tensor_rank_2_dim_4() -> TensorRank2<4, Current, Current> {
    TensorRank2::from([
        [3.0, 2.0, 3.0, 5.0],
        [6.0, 5.0, 2.0, 4.0],
        [4.0, 5.0, 0.0, 4.0],
        [4.0, 4.0, 1.0, 6.0],
    ])
}

fn get_diagonal_tensor_rank_2_dim_4() -> TensorRank2<4, Current, Current> {
    TensorRank2::from([
        [3.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 6.0],
    ])
}

fn get_other_tensor_rank_2_dim_9() -> TensorRank2<9, Current, Current> {
    TensorRank2::from([
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

fn get_other_tensor_rank_2_mul_tensor_rank_1_dim_4() -> TensorRank1<4, Current> {
    TensorRank1::from([51.0, 14.0, 22.0, 27.0])
}

fn get_other_tensor_rank_2_add_tensor_rank_2_dim_4() -> TensorRank2<4, Current, Current> {
    TensorRank2::from([
        [4.0, 6.0, 9.0, 11.0],
        [7.0, 10.0, 3.0, 4.0],
        [5.0, 8.0, 5.0, 4.0],
        [5.0, 8.0, 7.0, 6.0],
    ])
}

fn get_other_tensor_rank_2_sub_tensor_rank_2_dim_4() -> TensorRank2<4, Current, Current> {
    TensorRank2::from([
        [-2.0, 2.0, 3.0, 1.0],
        [-5.0, 0.0, -1.0, -4.0],
        [-3.0, -2.0, 5.0, -4.0],
        [-3.0, 0.0, 5.0, -6.0],
    ])
}

fn get_other_tensor_rank_2_mul_tensor_rank_2_dim_4() -> TensorRank2<4, Current, Current> {
    TensorRank2::from([
        [75.0, 76.0, 17.0, 81.0],
        [37.0, 32.0, 13.0, 29.0],
        [41.0, 42.0, 9.0, 37.0],
        [51.0, 52.0, 11.0, 45.0],
    ])
}

fn get_tensor_rank_1_list() -> TensorRank1List<3, Current, 8> {
    TensorRank1List::from([
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

fn get_tensor_rank_2_list_2d() -> TensorRank2List2D<3, Current, Current, 2, 2> {
    TensorRank2List2D::from([
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

fn get_tensor_rank_2_mul_tensor_rank_2_list_2d() -> TensorRank2List2D<3, Current, Current, 2, 2> {
    TensorRank2List2D::from([
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

fn get_tensor_rank_4() -> TensorRank4<3, Current, Current, Intermediate, Auxiliary> {
    TensorRank4::from([
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

fn get_tensor_rank_2_div_tensor_rank_4() -> TensorRank2<3, Intermediate, Auxiliary> {
    TensorRank2::from([
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
fn add_tensor_rank_2_to_self() -> Result<(), AssertionError> {
    Assert::eq(
        &(get_tensor_rank_2_dim_4() + get_other_tensor_rank_2_dim_4()),
        &get_other_tensor_rank_2_add_tensor_rank_2_dim_4(),
    )
}

#[test]
fn add_tensor_rank_2_ref_to_self() -> Result<(), AssertionError> {
    Assert::eq(
        &(get_tensor_rank_2_dim_4() + &get_other_tensor_rank_2_dim_4()),
        &get_other_tensor_rank_2_add_tensor_rank_2_dim_4(),
    )
}

#[test]
fn add_tensor_rank_2_to_self_ref() -> Result<(), AssertionError> {
    Assert::eq(
        &(&get_tensor_rank_2_dim_4() + get_other_tensor_rank_2_dim_4()),
        &get_other_tensor_rank_2_add_tensor_rank_2_dim_4(),
    )
}

#[test]
fn add_assign_tensor_rank_2() -> Result<(), AssertionError> {
    let mut tensor_rank_2 = get_tensor_rank_2_dim_4();
    tensor_rank_2 += get_other_tensor_rank_2_dim_4();
    Assert::eq(
        &tensor_rank_2,
        &get_other_tensor_rank_2_add_tensor_rank_2_dim_4(),
    )
}

#[test]
fn add_assign_tensor_rank_2_ref() -> Result<(), AssertionError> {
    let mut tensor_rank_2 = get_tensor_rank_2_dim_4();
    tensor_rank_2 += &get_other_tensor_rank_2_dim_4();
    Assert::eq(
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
fn div_tensor_rank_4_to_self() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        &(&get_tensor_rank_2_dim_3() / get_tensor_rank_4()),
        &get_tensor_rank_2_div_tensor_rank_4(),
    )
}

#[test]
fn div_tensor_rank_0_to_self() -> Result<(), AssertionError> {
    (get_tensor_rank_2_dim_4() / 3.3)
        .iter()
        .zip(get_array_dim_4().iter())
        .try_for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i.iter().zip(array_i.iter()).try_for_each(
                |(tensor_rank_2_ij, array_ij)| {
                    Assert::eq(
                        tensor_rank_2_ij,
                        &crate::math::Quantity::new(array_ij / 3.3),
                    )
                },
            )
        })?;
    Ok(())
}

#[test]
fn div_tensor_rank_0_to_self_ref() -> Result<(), AssertionError> {
    (&get_tensor_rank_2_dim_4() / 3.3)
        .iter()
        .zip(get_array_dim_4().iter())
        .try_for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i.iter().zip(array_i.iter()).try_for_each(
                |(tensor_rank_2_ij, array_ij)| {
                    Assert::eq(
                        tensor_rank_2_ij,
                        &crate::math::Quantity::new(array_ij / 3.3),
                    )
                },
            )
        })?;
    Ok(())
}

#[test]
#[allow(clippy::op_ref)]
fn div_tensor_rank_0_ref_to_self() -> Result<(), AssertionError> {
    (get_tensor_rank_2_dim_4() / &3.3)
        .iter()
        .zip(get_array_dim_4().iter())
        .try_for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i.iter().zip(array_i.iter()).try_for_each(
                |(tensor_rank_2_ij, array_ij)| {
                    Assert::eq(
                        tensor_rank_2_ij,
                        &crate::math::Quantity::new(array_ij / 3.3),
                    )
                },
            )
        })?;
    Ok(())
}

#[test]
#[allow(clippy::op_ref)]
fn div_tensor_rank_0_ref_to_self_ref() -> Result<(), AssertionError> {
    (&get_tensor_rank_2_dim_4() / &3.3)
        .iter()
        .zip(get_array_dim_4().iter())
        .try_for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i.iter().zip(array_i.iter()).try_for_each(
                |(tensor_rank_2_ij, array_ij)| {
                    Assert::eq(
                        tensor_rank_2_ij,
                        &crate::math::Quantity::new(array_ij / 3.3),
                    )
                },
            )
        })?;
    Ok(())
}

#[test]
fn div_assign_tensor_rank_0() -> Result<(), AssertionError> {
    let mut tensor_rank_2 = get_tensor_rank_2_dim_4();
    tensor_rank_2 /= 3.3;
    tensor_rank_2
        .iter()
        .zip(get_array_dim_4().iter())
        .try_for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i.iter().zip(array_i.iter()).try_for_each(
                |(tensor_rank_2_ij, array_ij)| {
                    Assert::eq(
                        tensor_rank_2_ij,
                        &crate::math::Quantity::new(array_ij / 3.3),
                    )
                },
            )
        })?;
    Ok(())
}

#[test]
fn div_assign_tensor_rank_0_ref() -> Result<(), AssertionError> {
    let mut tensor_rank_2 = get_tensor_rank_2_dim_4();
    tensor_rank_2 /= &3.3;
    tensor_rank_2
        .iter()
        .zip(get_array_dim_4().iter())
        .try_for_each(|(tensor_rank_2_i, array_i)| {
            tensor_rank_2_i.iter().zip(array_i.iter()).try_for_each(
                |(tensor_rank_2_ij, array_ij)| {
                    Assert::eq(
                        tensor_rank_2_ij,
                        &crate::math::Quantity::new(array_ij / 3.3),
                    )
                },
            )
        })?;
    Ok(())
}

#[test]
fn deviatoric_dim_2() -> Result<(), AssertionError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_2();
    let trace = tensor_rank_2.trace();
    let deviatoric_tensor_rank_2 = tensor_rank_2.deviatoric();
    Assert::zero(&deviatoric_tensor_rank_2.trace())?;
    Assert::eq(
        &deviatoric_tensor_rank_2,
        &(tensor_rank_2 - TensorRank2::identity() * (trace.value() / 2.0)),
    )
}

#[test]
fn deviatoric_dim_3() -> Result<(), AssertionError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_3();
    let trace = tensor_rank_2.trace();
    let deviatoric_tensor_rank_2 = tensor_rank_2.deviatoric();
    Assert::zero(&deviatoric_tensor_rank_2.trace())?;
    Assert::eq(
        &deviatoric_tensor_rank_2,
        &(tensor_rank_2 - TensorRank2::identity() * (trace.value() / 3.0)),
    )
}

#[test]
fn deviatoric_dim_4() -> Result<(), AssertionError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_4();
    let trace = tensor_rank_2.trace();
    let deviatoric_tensor_rank_2 = tensor_rank_2.deviatoric();
    Assert::zero(&deviatoric_tensor_rank_2.trace())?;
    Assert::eq(
        &deviatoric_tensor_rank_2,
        &(tensor_rank_2 - TensorRank2::identity() * (trace.value() / 4.0)),
    )
}

#[test]
fn deviatoric_dim_9() -> Result<(), AssertionError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_9();
    let trace = tensor_rank_2.trace();
    let deviatoric_tensor_rank_2 = tensor_rank_2.deviatoric();
    Assert::default().zero_within_tols(&deviatoric_tensor_rank_2.trace())?;
    Assert::eq(
        &deviatoric_tensor_rank_2,
        &(tensor_rank_2 - TensorRank2::identity() * (trace.value() / 9.0)),
    )
}

#[test]
fn deviatoric_and_trace_dim_2() -> Result<(), AssertionError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_2();
    let (deviatoric, trace) = tensor_rank_2.deviatoric_and_trace();
    Assert::eq(tensor_rank_2.trace(), &trace)?;
    Assert::eq(tensor_rank_2.deviatoric(), &deviatoric)
}

#[test]
fn deviatoric_and_trace_dim_3() -> Result<(), AssertionError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_3();
    let (deviatoric, trace) = tensor_rank_2.deviatoric_and_trace();
    Assert::eq(tensor_rank_2.trace(), &trace)?;
    Assert::eq(tensor_rank_2.deviatoric(), &deviatoric)
}

#[test]
fn deviatoric_and_trace_dim_4() -> Result<(), AssertionError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_4();
    let (deviatoric, trace) = tensor_rank_2.deviatoric_and_trace();
    Assert::eq(tensor_rank_2.trace(), &trace)?;
    Assert::eq(tensor_rank_2.deviatoric(), &deviatoric)
}

#[test]
fn deviatoric_and_trace_dim_9() -> Result<(), AssertionError> {
    let tensor_rank_2 = get_tensor_rank_2_dim_9();
    let (deviatoric, trace) = tensor_rank_2.deviatoric_and_trace();
    Assert::eq(tensor_rank_2.trace(), &trace)?;
    Assert::eq(tensor_rank_2.deviatoric(), &deviatoric)
}

#[test]
fn error() {
    let a = get_tensor_rank_1_a();
    let b = get_tensor_rank_1_b();
    assert_eq!(a.error_count(&a, ABS_TOL, REL_TOL), None);
    assert_eq!(a.error_count(&b, ABS_TOL, REL_TOL), Some(4));
}

#[test]
fn from_iter() {
    let into_iterator = get_tensor_rank_2_dim_4().0.into_iter();
    let tensor_rank_2 = TensorRank2::<4, Current, Current>::from_iter(get_tensor_rank_2_dim_4().0);
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
fn from_0_0_for_1_0() -> Result<(), AssertionError> {
    let tensor: TensorRank2<3, Current, Reference> =
        get_tensor_rank_2::<Reference, Reference>().into();
    Assert::eq(get_tensor_rank_2::<Current, Reference>(), &tensor)
}

#[test]
fn from_0_0_for_1_1() -> Result<(), AssertionError> {
    let tensor: TensorRank2<3, Current, Current> =
        get_tensor_rank_2::<Reference, Reference>().into();
    Assert::eq(get_tensor_rank_2::<Current, Current>(), &tensor)
}

#[test]
fn from_0_1_for_0_0() -> Result<(), AssertionError> {
    let tensor: TensorRank2<3, Reference, Reference> =
        get_tensor_rank_2::<Reference, Current>().into();
    Assert::eq(get_tensor_rank_2::<Reference, Reference>(), &tensor)
}

#[test]
fn from_1_0_for_0_0() -> Result<(), AssertionError> {
    let tensor: TensorRank2<3, Reference, Reference> =
        get_tensor_rank_2::<Current, Reference>().into();
    Assert::eq(get_tensor_rank_2::<Reference, Reference>(), &tensor)
}

#[test]
fn from_1_1_for_1_0() -> Result<(), AssertionError> {
    let tensor: TensorRank2<3, Current, Reference> = get_tensor_rank_2::<Current, Current>().into();
    Assert::eq(get_tensor_rank_2::<Current, Reference>(), &tensor)
}

#[test]
fn from_1_2_for_1_0() -> Result<(), AssertionError> {
    let tensor: TensorRank2<3, Current, Reference> =
        get_tensor_rank_2::<Current, Intermediate>().into();
    Assert::eq(get_tensor_rank_2::<Current, Reference>(), &tensor)
}

#[test]
fn full_contraction_dim_2() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        get_tensor_rank_2_dim_2().full_contraction(&get_other_tensor_rank_2_dim_2()),
        &70.0,
    )
}

#[test]
fn full_contraction_dim_3() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        get_tensor_rank_2_dim_3().full_contraction(&get_other_tensor_rank_2_dim_3()),
        &167.0,
    )
}

#[test]
fn full_contraction_dim_4() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        get_tensor_rank_2_dim_4().full_contraction(&get_other_tensor_rank_2_dim_4()),
        &137.0,
    )
}

#[test]
fn full_contraction_dim_9() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        get_tensor_rank_2_dim_9().full_contraction(&get_other_tensor_rank_2_dim_9()),
        &262.0,
    )
}

#[test]
fn identity() {
    TensorRank2::<9, Current, Current>::identity()
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
fn is_diagonal() {
    assert!(get_diagonal_tensor_rank_2_dim_4().is_diagonal())
}

#[test]
fn is_not_diagonal() {
    assert!(!get_other_tensor_rank_2_dim_4().is_diagonal())
}

#[test]
fn is_diagonal_identity() {
    assert!(TensorRank2::<3, Reference, Reference>::identity().is_diagonal())
}

#[test]
fn is_diagonal_zero() {
    assert!(TensorRank2::<4, Current, Current>::zero().is_diagonal())
}

#[test]
fn is_identity_dim_3() {
    assert!(TensorRank2::<3, Reference, Reference>::identity().is_identity())
}

#[test]
fn is_not_identity_dim_3() {
    assert!(!TensorRank2::<3, Reference, Reference>::zero().is_identity())
}

#[test]
fn is_identity_dim_4() {
    assert!(TensorRank2::<4, Current, Current>::identity().is_identity())
}

#[test]
fn is_not_identity_dim_4() {
    assert!(!get_diagonal_tensor_rank_2_dim_4().is_identity())
}

#[test]
fn is_zero_dim_3() {
    assert!(TensorRank2::<3, Reference, Reference>::zero().is_zero())
}

#[test]
fn is_not_zero_dim_3() {
    assert!(!TensorRank2::<3, Reference, Reference>::identity().is_zero())
}

#[test]
fn is_zero_dim_4() {
    assert!(TensorRank2::<4, Current, Current>::zero().is_zero())
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
fn into_vec() -> Result<(), AssertionError> {
    let vec: Vec<Vec<f64>> = get_tensor_rank_2_dim_4().into();
    get_tensor_rank_2_dim_4()
        .iter()
        .zip(vec.iter())
        .for_each(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .for_each(|(entry_a, entry_b)| assert_eq!(entry_a, entry_b))
        });
    Assert::eq(get_tensor_rank_2_dim_4(), &vec.into())
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
fn from() {
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
fn norm_dim_2() -> Result<(), AssertionError> {
    Assert::eq(
        get_tensor_rank_2_dim_2().norm(),
        &crate::math::Quantity::new(5.477_225_575_051_661),
    )
}

#[test]
fn norm_dim_3() -> Result<(), AssertionError> {
    Assert::eq(
        get_tensor_rank_2_dim_3().norm(),
        &crate::math::Quantity::new(16.881_943_016_134_134),
    )
}

#[test]
fn norm_dim_4() -> Result<(), AssertionError> {
    Assert::eq(
        get_tensor_rank_2_dim_4().norm(),
        &crate::math::Quantity::new(14.282_856_857_085_7),
    )
}

#[test]
fn norm_dim_9() -> Result<(), AssertionError> {
    Assert::eq(
        get_tensor_rank_2_dim_9().norm(),
        &crate::math::Quantity::new(20.736_441_353_327_72),
    )
}

#[test]
fn size() {
    assert_eq!(
        std::mem::size_of::<TensorRank2::<3, Current, Current>>(),
        std::mem::size_of::<[TensorRank1::<3, Current>; 3]>()
    )
}

#[test]
fn second_invariant() {
    assert_eq!(get_tensor_rank_2_dim_4().second_invariant(), 16.0);
}

#[test]
fn squared_trace_dim_2() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        get_tensor_rank_2_dim_2().squared_trace(),
        &crate::math::Quantity::new(29.0),
    )
}

#[test]
fn squared_trace_dim_3() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        get_tensor_rank_2_dim_3().squared_trace(),
        &crate::math::Quantity::new(258.0),
    )
}

#[test]
fn squared_trace_dim_4() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        get_tensor_rank_2_dim_4().squared_trace(),
        &crate::math::Quantity::new(89.0),
    )
}

#[test]
fn squared_trace_dim_9() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        get_tensor_rank_2_dim_9().squared_trace(),
        &crate::math::Quantity::new(308.0),
    )
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
fn trace_dim_2() -> Result<(), AssertionError> {
    Assert::eq(
        get_tensor_rank_2_dim_2().trace(),
        &crate::math::Quantity::new(5.0),
    )
}

#[test]
fn trace_dim_3() -> Result<(), AssertionError> {
    Assert::eq(
        get_tensor_rank_2_dim_3().trace(),
        &crate::math::Quantity::new(6.0),
    )
}

#[test]
fn trace_dim_4() -> Result<(), AssertionError> {
    Assert::eq(
        get_tensor_rank_2_dim_4().trace(),
        &crate::math::Quantity::new(11.0),
    )
}

#[test]
fn trace_dim_9() -> Result<(), AssertionError> {
    Assert::eq(
        get_tensor_rank_2_dim_9().trace(),
        &crate::math::Quantity::new(14.0),
    )
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
    TensorRank2::<2, Current, Current>::zero()
        .iter()
        .for_each(|tensor_rank_2_i| {
            tensor_rank_2_i
                .iter()
                .for_each(|tensor_rank_2_ij| assert_eq!(tensor_rank_2_ij, &0.0))
        });
}

#[test]
fn zero_dim_3() {
    TensorRank2::<3, Current, Current>::zero()
        .iter()
        .for_each(|tensor_rank_2_i| {
            tensor_rank_2_i
                .iter()
                .for_each(|tensor_rank_2_ij| assert_eq!(tensor_rank_2_ij, &0.0))
        });
}

#[test]
fn zero_dim_4() {
    TensorRank2::<4, Current, Current>::zero()
        .iter()
        .for_each(|tensor_rank_2_i| {
            tensor_rank_2_i
                .iter()
                .for_each(|tensor_rank_2_ij| assert_eq!(tensor_rank_2_ij, &0.0))
        });
}

#[test]
fn zero_dim_9() {
    TensorRank2::<9, Current, Current>::zero()
        .iter()
        .for_each(|tensor_rank_2_i| {
            tensor_rank_2_i
                .iter()
                .for_each(|tensor_rank_2_ij| assert_eq!(tensor_rank_2_ij, &0.0))
        });
}

#[test]
fn retain_from_filters_entries() {
    use super::super::{Jacobian, Vector};
    let retained = [true, false, true, false, true, true, false, true, false];
    let tensor = TensorRank2::<3, Current, Current>::from(get_array_dim_3());
    let vector = Jacobian::retain_from(tensor, &retained);
    let full = Vector::from(get_array_dim_3().as_flattened());
    let kept: Vec<usize> = (0..9).filter(|&index| retained[index]).collect();
    assert_eq!(vector.len(), kept.len());
    kept.iter()
        .enumerate()
        .for_each(|(p, &full_p)| assert_eq!(vector[p], full[full_p]))
}

#[test]
fn decrement_from_retained_skips_fixed_entries() {
    use super::super::{Solution, Vector};
    let retained = [true, false, true, false, true, true, false, true, false];
    let mut tensor = TensorRank2::<3, Current, Current>::from(get_array_dim_3());
    let decrement = Vector::from([1.0, 2.0, 3.0, 4.0, 5.0]);
    tensor.decrement_from_retained(&retained, &decrement);
    let full = get_array_dim_3();
    let mut taken = 0;
    (0..9).for_each(|index| {
        let before = full[index / 3][index % 3];
        let after = tensor[index / 3][index % 3];
        if retained[index] {
            assert_eq!(after, before - decrement[taken]);
            taken += 1
        } else {
            assert_eq!(after, before)
        }
    });
    assert_eq!(taken, 5)
}

#[test]
fn quadratic_form_matches_dense() {
    use super::super::{Hessian, SquareMatrix, Vector};
    let tensor = TensorRank2::<3, Current, Current>::from(get_array_dim_3());
    let vector = Vector::from([1.0, -2.0, 3.0]);
    let form = tensor.quadratic_form(&vector);
    let mut dense = SquareMatrix::zero(3);
    tensor.fill_into(&mut dense);
    Assert::default()
        .eq_within_tols(form, &(&vector * (dense * &vector)))
        .unwrap()
}

/// Nothing overrides the default here, so this covers reaching every entry by
/// position rather than by walking what is stored.
#[test]
fn times_matches_dense() {
    use super::super::{Hessian, SquareMatrix, Vector};
    let tensor = TensorRank2::<3, Current, Current>::from(get_array_dim_3());
    let vector = Vector::from([1.0, -2.0, 3.0]);
    let product = tensor.times(&vector);
    let mut dense = SquareMatrix::zero(3);
    tensor.fill_into(&mut dense);
    Assert::default()
        .eq_within_tols(&product, &(dense * &vector))
        .unwrap()
}
