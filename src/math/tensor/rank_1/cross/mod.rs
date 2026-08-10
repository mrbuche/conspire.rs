#[cfg(test)]
mod test;

use super::TensorRank1;

/// The cross product of two rank-1 tensors.
pub trait CrossProduct<T> {
    /// The specific return type.
    type Output;
    /// Returns the cross product with another rank-1 tensor.
    fn cross(self, other: T) -> Self::Output;
}

impl<I, U> CrossProduct<TensorRank1<3, I, U>> for &TensorRank1<3, I, U> {
    type Output = TensorRank1<3, I, U>;
    fn cross(self, other: TensorRank1<3, I, U>) -> Self::Output {
        TensorRank1::const_from([
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ])
    }
}

impl<I, U> CrossProduct<Self> for &TensorRank1<3, I, U> {
    type Output = TensorRank1<3, I, U>;
    fn cross(self, other: Self) -> Self::Output {
        TensorRank1::const_from([
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ])
    }
}
