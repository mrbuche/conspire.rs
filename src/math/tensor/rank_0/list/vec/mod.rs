use crate::math::{TensorRank0List, TensorVector};

pub type TensorRank0ListVec<const W: usize> = TensorVector<TensorRank0List<W>>;
