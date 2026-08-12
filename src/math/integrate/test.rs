use crate::math::{Quantity, TensorRank0, Time};
use std::array::from_fn;
pub const LENGTH: usize = 33;

pub fn zero_to_one<const W: usize>() -> [Quantity<Time>; W] {
    from_fn(|i| Quantity::new((i as TensorRank0) / ((W - 1) as TensorRank0)))
}
