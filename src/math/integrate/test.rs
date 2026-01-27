use super::IntegrationError;
use crate::math::{TensorArray, TensorRank0, TensorRank0List};

pub const LENGTH: usize = 33;

pub fn zero_to_one<const W: usize>() -> [TensorRank0; W] {
    (0..W)
        .map(|i| (i as TensorRank0) / ((W - 1) as TensorRank0))
        .collect::<TensorRank0List<W>>()
        .as_array()
}

#[test]
fn debug() {
    let _ = format!("{:?}", IntegrationError::InitialTimeNotLessThanFinalTime);
    let _ = format!("{:?}", IntegrationError::LengthTimeLessThanTwo);
}

#[test]
fn display() {
    let _ = format!("{}", IntegrationError::InitialTimeNotLessThanFinalTime);
    let _ = format!("{}", IntegrationError::LengthTimeLessThanTwo);
}
