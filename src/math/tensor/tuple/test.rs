use super::TensorTuple;
use crate::math::{
    Current, Jacobian, Quantity, Solution, Tensor, TensorRank1, TensorRank2, Vector,
};
use crate::units::Dimensionless;

type Fp = TensorRank2<3, Current, Current>;

fn fp() -> Fp {
    Fp::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
}

#[test]
fn mixed_tuple_fill_into_lays_out_head_then_tail() {
    let tuple = TensorTuple::from((fp(), Quantity::<Dimensionless>::new(10.0)));
    assert_eq!(tuple.size(), 10);
    let mut vector = Vector::zero(10);
    tuple.fill_into(&mut vector);
    (0..10).for_each(|i| assert_eq!(vector[i], (i + 1) as f64));
}

#[test]
fn mixed_tuple_decrement_from_is_the_inverse_of_fill_into() {
    let mut tuple = TensorTuple::from((fp(), Quantity::<Dimensionless>::new(10.0)));
    let mut vector = Vector::zero(10);
    tuple.fill_into(&mut vector);
    tuple.decrement_from(&vector);
    let mut zeroed = Vector::zero(10);
    tuple.fill_into(&mut zeroed);
    zeroed.iter().for_each(|&entry| assert_eq!(entry, 0.0));
}

#[test]
fn nested_mixed_tuple_round_trips() {
    let mut tuple = TensorTuple::from((
        fp(),
        TensorTuple::from((
            TensorRank1::<3, Current>::from([10.0, 11.0, 12.0]),
            Quantity::<Dimensionless>::new(13.0),
        )),
    ));
    assert_eq!(tuple.size(), 13);
    let mut vector = Vector::zero(13);
    tuple.fill_into(&mut vector);
    (0..13).for_each(|i| assert_eq!(vector[i], (i + 1) as f64));
    tuple.decrement_from(&vector);
    let mut zeroed = Vector::zero(13);
    tuple.fill_into(&mut zeroed);
    zeroed.iter().for_each(|&entry| assert_eq!(entry, 0.0));
}
