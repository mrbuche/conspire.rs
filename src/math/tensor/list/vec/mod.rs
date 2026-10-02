use crate::math::{TensorList, TensorVector};
/// A vector of lists of tensors.
pub type TensorListVec<T, const N: usize> = TensorVector<TensorList<T, N>>;
