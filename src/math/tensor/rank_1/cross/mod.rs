#[cfg(test)]
mod test;

use super::TensorRank1;

/// The cross product of two rank-1 tensors.
pub trait CrossProduct<T> {
    type Output;
    /// Returns the cross product with another rank-1 tensor.
    fn cross(self, other: T) -> Self::Output;
}

impl<const I: usize> CrossProduct<TensorRank1<3, I>> for &TensorRank1<3, I> {
    type Output = TensorRank1<3, I>;
    fn cross(self, other: TensorRank1<3, I>) -> Self::Output {
        TensorRank1::const_from([
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ])
    }
}

impl<const I: usize> CrossProduct<Self> for &TensorRank1<3, I> {
    type Output = TensorRank1<3, I>;
    fn cross(self, other: Self) -> Self::Output {
        TensorRank1::const_from([
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ])
    }
}
