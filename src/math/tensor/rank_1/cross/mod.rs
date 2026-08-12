#[cfg(test)]
mod test;

use super::TensorRank1;
use crate::math::unit::UnitMul;

/// The cross product of two rank-1 tensors.
pub trait CrossProduct<T> {
    /// The specific return type.
    type Output;
    /// Returns the cross product with another rank-1 tensor.
    fn cross(self, other: T) -> Self::Output;
}

impl<I, U, V> CrossProduct<TensorRank1<3, I, V>> for &TensorRank1<3, I, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1<3, I, <U as UnitMul<V>>::Output>;
    fn cross(self, other: TensorRank1<3, I, V>) -> Self::Output {
        TensorRank1::const_from([
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ])
    }
}

impl<'a, I, U, V> CrossProduct<&'a TensorRank1<3, I, V>> for &TensorRank1<3, I, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1<3, I, <U as UnitMul<V>>::Output>;
    fn cross(self, other: &'a TensorRank1<3, I, V>) -> Self::Output {
        TensorRank1::const_from([
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ])
    }
}
