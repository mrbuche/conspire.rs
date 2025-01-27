use super::IntegrationError;
use crate::math::{test::TestError, TensorArray, TensorRank0, TensorRank0List};
use std::f64::consts::TAU;

pub fn zero_to_tau<const W: usize>() -> [TensorRank0; W] {
    (0..W)
        .map(|i| TAU * (i as TensorRank0) / ((W - 1) as TensorRank0))
        .collect::<TensorRank0List<W>>()
        .as_array()
}

impl From<IntegrationError> for TestError {
    fn from(error: IntegrationError) -> TestError {
        TestError {
            message: format!("{}", error),
        }
    }
}
