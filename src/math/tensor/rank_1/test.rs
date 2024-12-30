use super::{Tensor, TensorArray, TensorRank0, TensorRank1};
use crate::{
    math::test::{assert_eq, ErrorTensor, TestError},
    ABS_TOL, REL_TOL,
};

fn get_array() -> [TensorRank0; 4] {
    [1.0, 2.0, 3.0, 4.0]
}

fn get_tensor_rank_1() -> TensorRank1<4, 1> {
    TensorRank1::new(get_array())
}

fn get_tensor_rank_1_a() -> TensorRank1<4, 1> {
    TensorRank1::new([5.0, 7.0, 6.0, 8.0])
}

fn get_tensor_rank_1_b() -> TensorRank1<3, 1> {
    TensorRank1::new([7.0, 2.0, 3.0])
}

fn get_tensor_rank_1_c() -> TensorRank1<3, 1> {
    TensorRank1::new([4.0, 5.0, 6.0])
}

fn get_tensor_rank_1_b_cross_c() -> TensorRank1<3, 1> {
    TensorRank1::new([-3.0, -30.0, 27.0])
}

fn get_tensor_rank_1_add_tensor_rank_1_a() -> TensorRank1<4, 1> {
    TensorRank1::new([6.0, 9.0, 9.0, 12.0])
}

fn get_tensor_rank_1_mul_tensor_rank_1_a() -> TensorRank0 {
    69.0
}

fn get_tensor_rank_1_sub_tensor_rank_1_a() -> TensorRank1<4, 1> {
    TensorRank1::new([-4.0, -5.0, -3.0, -4.0])
}

#[test]
fn add_tensor_rank_1_to_self() -> Result<(), TestError> {
    assert_eq(
        &(get_tensor_rank_1() + get_tensor_rank_1_a()),
        &get_tensor_rank_1_add_tensor_rank_1_a(),
    )
}

#[test]
fn add_tensor_rank_1_ref_to_self() -> Result<(), TestError> {
    assert_eq(
        &(get_tensor_rank_1() + &get_tensor_rank_1_a()),
        &get_tensor_rank_1_add_tensor_rank_1_a(),
    )
}

#[test]
fn add_tensor_rank_1_to_self_ref() -> Result<(), TestError> {
    assert_eq(
        &(&get_tensor_rank_1() + get_tensor_rank_1_a()),
        &get_tensor_rank_1_add_tensor_rank_1_a(),
    )
}

#[test]
fn add_assign_tensor_rank_1() -> Result<(), TestError> {
    let mut tensor_rank_1 = get_tensor_rank_1();
    tensor_rank_1 += get_tensor_rank_1_a();
    assert_eq(&tensor_rank_1, &get_tensor_rank_1_add_tensor_rank_1_a())
}

#[test]
fn add_assign_tensor_rank_1_ref() -> Result<(), TestError> {
    let mut tensor_rank_1 = get_tensor_rank_1();
    tensor_rank_1 += &get_tensor_rank_1_a();
    assert_eq(&tensor_rank_1, &get_tensor_rank_1_add_tensor_rank_1_a())
}

#[test]
fn as_array() {
    assert_eq!(get_tensor_rank_1().as_array(), get_array())
}

#[test]
fn cross() -> Result<(), TestError> {
    assert_eq(
        &(get_tensor_rank_1_b().cross(&get_tensor_rank_1_c())),
        &get_tensor_rank_1_b_cross_c(),
    )
}

#[test]
#[should_panic]
fn cross_panic() {
    get_tensor_rank_1().cross(&get_tensor_rank_1_a());
}

#[test]
fn div_tensor_rank_0_to_self() -> Result<(), TestError> {
    (get_tensor_rank_1() / 3.3)
        .iter()
        .zip(get_array().iter())
        .try_for_each(|(tensor_rank_1_i, array_i)| assert_eq(tensor_rank_1_i, &(array_i / 3.3)))
}

#[test]
fn div_tensor_rank_0_to_self_ref() -> Result<(), TestError> {
    (&get_tensor_rank_1() / 3.3)
        .iter()
        .zip(get_array().iter())
        .try_for_each(|(tensor_rank_1_i, array_i)| assert_eq(tensor_rank_1_i, &(array_i / 3.3)))
}

#[test]
#[allow(clippy::op_ref)]
fn div_tensor_rank_0_ref_to_self() -> Result<(), TestError> {
    (get_tensor_rank_1() / &3.3)
        .iter()
        .zip(get_array().iter())
        .try_for_each(|(tensor_rank_1_i, array_i)| assert_eq(tensor_rank_1_i, &(array_i / 3.3)))
}

#[test]
#[allow(clippy::op_ref)]
fn div_tensor_rank_0_ref_to_self_ref() -> Result<(), TestError> {
    (&get_tensor_rank_1() / &3.3)
        .iter()
        .zip(get_array().iter())
        .try_for_each(|(tensor_rank_1_i, array_i)| assert_eq(tensor_rank_1_i, &(array_i / 3.3)))
}

#[test]
fn div_assign_tensor_rank_0() -> Result<(), TestError> {
    let mut tensor_rank_1 = get_tensor_rank_1();
    tensor_rank_1 /= 3.3;
    let result = tensor_rank_1
        .iter()
        .zip(get_array().iter())
        .try_for_each(|(tensor_rank_1_i, array_i)| assert_eq(tensor_rank_1_i, &(array_i / 3.3)));
    result
}

#[test]
fn div_assign_tensor_rank_0_ref() -> Result<(), TestError> {
    let mut tensor_rank_1 = get_tensor_rank_1();
    tensor_rank_1 /= &3.3;
    let result = tensor_rank_1
        .iter()
        .zip(get_array().iter())
        .try_for_each(|(tensor_rank_1_i, array_i)| assert_eq(tensor_rank_1_i, &(array_i / 3.3)));
    result
}

#[test]
fn error() {
    let b = get_tensor_rank_1_b();
    let c = get_tensor_rank_1_c();
    assert_eq!(b.error(&b, &ABS_TOL, &REL_TOL), None);
    assert_eq!(b.error(&c, &ABS_TOL, &REL_TOL), Some(3));
}

#[test]
fn from_iter() {
    let into_iterator = (0..8).map(|x| x as TensorRank0);
    let tensor_rank_1 = TensorRank1::<8, 1>::from_iter(into_iterator.clone());
    tensor_rank_1
        .iter()
        .zip(into_iterator)
        .for_each(|(tensor_rank_1_i, value_i)| assert_eq!(tensor_rank_1_i, &value_i));
}

#[test]
fn iter() {
    get_tensor_rank_1()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_i, array_i)| assert_eq!(tensor_rank_1_i, array_i));
}

#[test]
fn iter_mut() {
    get_tensor_rank_1()
        .iter_mut()
        .zip(get_array().iter_mut())
        .for_each(|(tensor_rank_1_i, array_i)| assert_eq!(tensor_rank_1_i, array_i));
}

#[test]
fn mul_tensor_rank_0_to_self() {
    (get_tensor_rank_1() * 3.3)
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_i, array_i)| assert_eq!(tensor_rank_1_i, &(array_i * 3.3)));
}

#[test]
fn mul_tensor_rank_0_to_self_ref() {
    (&get_tensor_rank_1() * 3.3)
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_i, array_i)| assert_eq!(tensor_rank_1_i, &(array_i * 3.3)));
}

#[test]
#[allow(clippy::op_ref)]
fn mul_tensor_rank_0_ref_to_self() {
    (get_tensor_rank_1() * &3.3)
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_i, array_i)| assert_eq!(tensor_rank_1_i, &(array_i * 3.3)));
}

#[test]
#[allow(clippy::op_ref)]
fn mul_tensor_rank_0_ref_to_self_ref() {
    (&get_tensor_rank_1() * &3.3)
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_i, array_i)| assert_eq!(tensor_rank_1_i, &(array_i * 3.3)));
}

#[test]
fn mul_assign_tensor_rank_0() {
    let mut tensor_rank_1 = get_tensor_rank_1();
    tensor_rank_1 *= 3.3;
    tensor_rank_1
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_i, array_i)| assert_eq!(tensor_rank_1_i, &(array_i * 3.3)));
}

#[test]
fn mul_assign_tensor_rank_0_ref() {
    let mut tensor_rank_1 = get_tensor_rank_1();
    tensor_rank_1 *= &3.3;
    tensor_rank_1
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_i, array_i)| assert_eq!(tensor_rank_1_i, &(array_i * 3.3)));
}

#[test]
fn mul_tensor_rank_1_to_self() {
    assert_eq!(
        get_tensor_rank_1() * get_tensor_rank_1_a(),
        get_tensor_rank_1_mul_tensor_rank_1_a()
    )
}

#[test]
fn mul_tensor_rank_1_ref_to_self() {
    assert_eq!(
        get_tensor_rank_1() * &get_tensor_rank_1_a(),
        get_tensor_rank_1_mul_tensor_rank_1_a()
    )
}

#[test]
fn mul_tensor_rank_1_to_self_ref() {
    assert_eq!(
        &get_tensor_rank_1() * get_tensor_rank_1_a(),
        get_tensor_rank_1_mul_tensor_rank_1_a()
    )
}

#[test]
fn mul_tensor_rank_1_ref_to_self_ref() {
    assert_eq!(
        &get_tensor_rank_1() * &get_tensor_rank_1_a(),
        get_tensor_rank_1_mul_tensor_rank_1_a()
    )
}

#[test]
fn new() {
    get_tensor_rank_1()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_i, array_i)| assert_eq!(tensor_rank_1_i, array_i));
}

#[test]
fn norm() {
    assert_eq!(get_tensor_rank_1().norm(), 5.477_225_575_051_661);
}

#[test]
fn normalized() {
    assert_eq!(get_tensor_rank_1().normalized().norm(), 0.9999999999999999);
}

#[test]
fn size() {
    assert_eq!(
        std::mem::size_of::<TensorRank1::<3, 1>>(),
        std::mem::size_of::<[TensorRank0; 3]>()
    )
}

#[test]
fn sub_tensor_rank_1_to_self() {
    (get_tensor_rank_1() - get_tensor_rank_1_a())
        .iter()
        .zip(get_tensor_rank_1_sub_tensor_rank_1_a().iter())
        .for_each(|(tensor_rank_1_i, sub_tensor_rank_1_i)| {
            assert_eq!(tensor_rank_1_i, sub_tensor_rank_1_i)
        });
}

#[test]
fn sub_tensor_rank_1_ref_to_self() {
    (get_tensor_rank_1() - &get_tensor_rank_1_a())
        .iter()
        .zip(get_tensor_rank_1_sub_tensor_rank_1_a().iter())
        .for_each(|(tensor_rank_1_i, sub_tensor_rank_1_i)| {
            assert_eq!(tensor_rank_1_i, sub_tensor_rank_1_i)
        });
}

#[test]
fn sub_tensor_rank_1_to_self_ref() {
    (&get_tensor_rank_1() - get_tensor_rank_1_a())
        .iter()
        .zip(get_tensor_rank_1_sub_tensor_rank_1_a().iter())
        .for_each(|(tensor_rank_1_i, sub_tensor_rank_1_i)| {
            assert_eq!(tensor_rank_1_i, sub_tensor_rank_1_i)
        });
}

#[test]
fn sub_tensor_rank_1_ref_to_self_ref() {
    (&get_tensor_rank_1() - &get_tensor_rank_1_a())
        .iter()
        .zip(get_tensor_rank_1_sub_tensor_rank_1_a().iter())
        .for_each(|(tensor_rank_1_i, sub_tensor_rank_1_i)| {
            assert_eq!(tensor_rank_1_i, sub_tensor_rank_1_i)
        });
}

#[test]
fn sub_assign_tensor_rank_1() {
    let mut tensor_rank_1 = get_tensor_rank_1();
    tensor_rank_1 -= get_tensor_rank_1_a();
    tensor_rank_1
        .iter()
        .zip(get_tensor_rank_1_sub_tensor_rank_1_a().iter())
        .for_each(|(tensor_rank_1_i, sub_tensor_rank_1_i)| {
            assert_eq!(tensor_rank_1_i, sub_tensor_rank_1_i)
        });
}

#[test]
fn sub_assign_tensor_rank_1_ref() {
    let mut tensor_rank_1 = get_tensor_rank_1();
    tensor_rank_1 -= &get_tensor_rank_1_a();
    tensor_rank_1
        .iter()
        .zip(get_tensor_rank_1_sub_tensor_rank_1_a().iter())
        .for_each(|(tensor_rank_1_i, sub_tensor_rank_1_i)| {
            assert_eq!(tensor_rank_1_i, sub_tensor_rank_1_i)
        });
}

#[test]
fn zero() {
    TensorRank1::<8, 1>::zero()
        .iter()
        .for_each(|tensor_rank_1_i| assert_eq!(tensor_rank_1_i, &0.0));
}
