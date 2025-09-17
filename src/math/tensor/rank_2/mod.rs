#[cfg(test)]
mod test;

pub mod list;
pub mod list_2d;
pub mod vec;
pub mod vec_2d;

use std::{
    array::from_fn,
    f64::consts::TAU,
    fmt::{self, Display, Formatter},
    mem::transmute,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use super::{
    super::assert_eq_within_tols,
    Hessian, Jacobian, Rank2, Solution, SquareMatrix, Tensor, TensorArray, Vector,
    rank_0::{TensorRank0, list::TensorRank0List},
    rank_1::{
        TensorRank1, list::TensorRank1List, tensor_rank_1, vec::TensorRank1Vec,
        zero as tensor_rank_1_zero,
    },
    rank_4::TensorRank4,
};
use crate::ABS_TOL;
use list_2d::TensorRank2List2D;
use vec_2d::TensorRank2Vec2D;

#[cfg(test)]
use super::test::ErrorTensor;

/// A *d*-dimensional tensor of rank 2.
///
/// `D` is the dimension, `I`, `J` are the configurations.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorRank2<const D: usize, const I: usize, const J: usize>([TensorRank1<D, J>; D]);

pub const fn tensor_rank_2<const D: usize, const I: usize, const J: usize>(
    array: [TensorRank1<D, J>; D],
) -> TensorRank2<D, I, J> {
    TensorRank2(array)
}

pub const fn get_levi_civita_parts<const I: usize, const J: usize>() -> [TensorRank2<3, I, J>; 3] {
    [
        TensorRank2([
            tensor_rank_1_zero(),
            tensor_rank_1([0.0, 0.0, 1.0]),
            tensor_rank_1([0.0, -1.0, 0.0]),
        ]),
        TensorRank2([
            tensor_rank_1([0.0, 0.0, -1.0]),
            tensor_rank_1_zero(),
            tensor_rank_1([1.0, 0.0, 0.0]),
        ]),
        TensorRank2([
            tensor_rank_1([0.0, 1.0, 0.0]),
            tensor_rank_1([-1.0, 0.0, 0.0]),
            tensor_rank_1_zero(),
        ]),
    ]
}

pub const fn get_identity_1010_parts_1<const I: usize, const J: usize>() -> [TensorRank2<3, I, J>; 3]
{
    [
        TensorRank2([
            tensor_rank_1([1.0, 0.0, 0.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
        ]),
        TensorRank2([
            tensor_rank_1([0.0, 1.0, 0.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
        ]),
        TensorRank2([
            tensor_rank_1([0.0, 0.0, 1.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
        ]),
    ]
}

pub const fn get_identity_1010_parts_2<const I: usize, const J: usize>() -> [TensorRank2<3, I, J>; 3]
{
    [
        TensorRank2([
            tensor_rank_1_zero(),
            tensor_rank_1([1.0, 0.0, 0.0]),
            tensor_rank_1_zero(),
        ]),
        TensorRank2([
            tensor_rank_1_zero(),
            tensor_rank_1([0.0, 1.0, 0.0]),
            tensor_rank_1_zero(),
        ]),
        TensorRank2([
            tensor_rank_1_zero(),
            tensor_rank_1([0.0, 0.0, 1.0]),
            tensor_rank_1_zero(),
        ]),
    ]
}

pub const fn get_identity_1010_parts_3<const I: usize, const J: usize>() -> [TensorRank2<3, I, J>; 3]
{
    [
        TensorRank2([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1([1.0, 0.0, 0.0]),
        ]),
        TensorRank2([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1([0.0, 1.0, 0.0]),
        ]),
        TensorRank2([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1([0.0, 0.0, 1.0]),
        ]),
    ]
}

pub const IDENTITY: TensorRank2<3, 1, 1> = TensorRank2([
    tensor_rank_1([1.0, 0.0, 0.0]),
    tensor_rank_1([0.0, 1.0, 0.0]),
    tensor_rank_1([0.0, 0.0, 1.0]),
]);

pub const IDENTITY_00: TensorRank2<3, 0, 0> = TensorRank2([
    tensor_rank_1([1.0, 0.0, 0.0]),
    tensor_rank_1([0.0, 1.0, 0.0]),
    tensor_rank_1([0.0, 0.0, 1.0]),
]);

pub const IDENTITY_10: TensorRank2<3, 1, 0> = TensorRank2([
    tensor_rank_1([1.0, 0.0, 0.0]),
    tensor_rank_1([0.0, 1.0, 0.0]),
    tensor_rank_1([0.0, 0.0, 1.0]),
]);

pub const ZERO: TensorRank2<3, 1, 1> = TensorRank2([
    tensor_rank_1_zero(),
    tensor_rank_1_zero(),
    tensor_rank_1_zero(),
]);

pub const ZERO_10: TensorRank2<3, 1, 0> = TensorRank2([
    tensor_rank_1_zero(),
    tensor_rank_1_zero(),
    tensor_rank_1_zero(),
]);

impl<const D: usize, const I: usize, const J: usize> From<Vec<Vec<TensorRank0>>>
    for TensorRank2<D, I, J>
{
    fn from(vec: Vec<Vec<TensorRank0>>) -> Self {
        assert_eq!(vec.len(), D);
        vec.iter().for_each(|entry| assert_eq!(entry.len(), D));
        vec.into_iter()
            .map(|entry| entry.into_iter().collect())
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> From<TensorRank2<D, I, J>>
    for Vec<Vec<TensorRank0>>
{
    fn from(tensor: TensorRank2<D, I, J>) -> Self {
        tensor
            .iter()
            .map(|entry| entry.iter().copied().collect())
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Display for TensorRank2<D, I, J> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "[")?;
        self.iter()
            .enumerate()
            .try_for_each(|(i, row)| write!(f, "{row},\n\x1B[u\x1B[{}B", i + 1))?;
        write!(f, "\x1B[u\x1B[1A\x1B[{}C]", 16 * D)
    }
}

#[cfg(test)]
impl<const D: usize, const I: usize, const J: usize> ErrorTensor for TensorRank2<D, I, J> {
    fn error_fd(&self, comparator: &Self, epsilon: &TensorRank0) -> Option<(bool, usize)> {
        let error_count = self
            .iter()
            .zip(comparator.iter())
            .map(|(self_i, comparator_i)| {
                self_i
                    .iter()
                    .zip(comparator_i.iter())
                    .filter(|&(&self_ij, &comparator_ij)| {
                        &(self_ij / comparator_ij - 1.0).abs() >= epsilon
                            && (&self_ij.abs() >= epsilon || &comparator_ij.abs() >= epsilon)
                    })
                    .count()
            })
            .sum();
        if error_count > 0 {
            Some((true, error_count))
        } else {
            None
        }
    }
}

impl<const D: usize, const I: usize, const J: usize> TensorRank2<D, I, J> {
    /// Returns a raw pointer to the slice’s buffer.
    pub const fn as_ptr(&self) -> *const TensorRank1<D, J> {
        self.0.as_ptr()
    }
    /// Returns the rank-2 tensor reshaped as a rank-1 tensor.
    pub fn as_tensor_rank_1(&self) -> TensorRank1<9, 88> {
        assert_eq!(D, 3);
        let mut tensor_rank_1 = TensorRank1::<9, 88>::zero();
        self.iter().enumerate().for_each(|(i, self_i)| {
            self_i
                .iter()
                .enumerate()
                .for_each(|(j, self_ij)| tensor_rank_1[3 * i + j] = *self_ij)
        });
        tensor_rank_1
    }
    /// Returns the determinant of the rank-2 tensor.
    pub fn determinant(&self) -> TensorRank0 {
        if D == 2 {
            self[0][0] * self[1][1] - self[0][1] * self[1][0]
        } else if D == 3 {
            let c_00 = self[1][1] * self[2][2] - self[1][2] * self[2][1];
            let c_10 = self[1][2] * self[2][0] - self[1][0] * self[2][2];
            let c_20 = self[1][0] * self[2][1] - self[1][1] * self[2][0];
            self[0][0] * c_00 + self[0][1] * c_10 + self[0][2] * c_20
        } else if D == 4 {
            let s0 = self[0][0] * self[1][1] - self[0][1] * self[1][0];
            let s1 = self[0][0] * self[1][2] - self[0][2] * self[1][0];
            let s2 = self[0][0] * self[1][3] - self[0][3] * self[1][0];
            let s3 = self[0][1] * self[1][2] - self[0][2] * self[1][1];
            let s4 = self[0][1] * self[1][3] - self[0][3] * self[1][1];
            let s5 = self[0][2] * self[1][3] - self[0][3] * self[1][2];
            let c5 = self[2][2] * self[3][3] - self[2][3] * self[3][2];
            let c4 = self[2][1] * self[3][3] - self[2][3] * self[3][1];
            let c3 = self[2][1] * self[3][2] - self[2][2] * self[3][1];
            let c2 = self[2][0] * self[3][3] - self[2][3] * self[3][0];
            let c1 = self[2][0] * self[3][2] - self[2][2] * self[3][0];
            let c0 = self[2][0] * self[3][1] - self[2][1] * self[3][0];
            s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0
        } else {
            let (tensor_l, tensor_u) = self.lu_decomposition();
            tensor_l
                .iter()
                .enumerate()
                .zip(tensor_u.iter())
                .map(|((i, tensor_l_i), tensor_u_i)| tensor_l_i[i] * tensor_u_i[i])
                .product()
        }
    }
    /// Returns a rank-2 tensor constructed from a dyad of the given vectors.
    pub fn dyad(vector_a: &TensorRank1<D, I>, vector_b: &TensorRank1<D, J>) -> Self {
        vector_a
            .iter()
            .map(|vector_a_i| {
                vector_b
                    .iter()
                    .map(|vector_b_j| vector_a_i * vector_b_j)
                    .collect()
            })
            .collect()
    }
    /// Returns the inverse of the rank-2 tensor.
    pub fn inverse(&self) -> TensorRank2<D, J, I> {
        if D == 2 {
            let mut adjugate = TensorRank2::<D, J, I>::zero();
            adjugate[0][0] = self[1][1];
            adjugate[0][1] = -self[0][1];
            adjugate[1][0] = -self[1][0];
            adjugate[1][1] = self[0][0];
            adjugate / self.determinant()
        } else if D == 3 {
            let mut adjugate = TensorRank2::<D, J, I>::zero();
            let c_00 = self[1][1] * self[2][2] - self[1][2] * self[2][1];
            let c_10 = self[1][2] * self[2][0] - self[1][0] * self[2][2];
            let c_20 = self[1][0] * self[2][1] - self[1][1] * self[2][0];
            adjugate[0][0] = c_00;
            adjugate[0][1] = self[0][2] * self[2][1] - self[0][1] * self[2][2];
            adjugate[0][2] = self[0][1] * self[1][2] - self[0][2] * self[1][1];
            adjugate[1][0] = c_10;
            adjugate[1][1] = self[0][0] * self[2][2] - self[0][2] * self[2][0];
            adjugate[1][2] = self[0][2] * self[1][0] - self[0][0] * self[1][2];
            adjugate[2][0] = c_20;
            adjugate[2][1] = self[0][1] * self[2][0] - self[0][0] * self[2][1];
            adjugate[2][2] = self[0][0] * self[1][1] - self[0][1] * self[1][0];
            adjugate / (self[0][0] * c_00 + self[0][1] * c_10 + self[0][2] * c_20)
        } else if D == 4 {
            let mut adjugate = TensorRank2::<D, J, I>::zero();
            let s0 = self[0][0] * self[1][1] - self[0][1] * self[1][0];
            let s1 = self[0][0] * self[1][2] - self[0][2] * self[1][0];
            let s2 = self[0][0] * self[1][3] - self[0][3] * self[1][0];
            let s3 = self[0][1] * self[1][2] - self[0][2] * self[1][1];
            let s4 = self[0][1] * self[1][3] - self[0][3] * self[1][1];
            let s5 = self[0][2] * self[1][3] - self[0][3] * self[1][2];
            let c5 = self[2][2] * self[3][3] - self[2][3] * self[3][2];
            let c4 = self[2][1] * self[3][3] - self[2][3] * self[3][1];
            let c3 = self[2][1] * self[3][2] - self[2][2] * self[3][1];
            let c2 = self[2][0] * self[3][3] - self[2][3] * self[3][0];
            let c1 = self[2][0] * self[3][2] - self[2][2] * self[3][0];
            let c0 = self[2][0] * self[3][1] - self[2][1] * self[3][0];
            adjugate[0][0] = self[1][1] * c5 - self[1][2] * c4 + self[1][3] * c3;
            adjugate[0][1] = self[0][2] * c4 - self[0][1] * c5 - self[0][3] * c3;
            adjugate[0][2] = self[3][1] * s5 - self[3][2] * s4 + self[3][3] * s3;
            adjugate[0][3] = self[2][2] * s4 - self[2][1] * s5 - self[2][3] * s3;
            adjugate[1][0] = self[1][2] * c2 - self[1][0] * c5 - self[1][3] * c1;
            adjugate[1][1] = self[0][0] * c5 - self[0][2] * c2 + self[0][3] * c1;
            adjugate[1][2] = self[3][2] * s2 - self[3][0] * s5 - self[3][3] * s1;
            adjugate[1][3] = self[2][0] * s5 - self[2][2] * s2 + self[2][3] * s1;
            adjugate[2][0] = self[1][0] * c4 - self[1][1] * c2 + self[1][3] * c0;
            adjugate[2][1] = self[0][1] * c2 - self[0][0] * c4 - self[0][3] * c0;
            adjugate[2][2] = self[3][0] * s4 - self[3][1] * s2 + self[3][3] * s0;
            adjugate[2][3] = self[2][1] * s2 - self[2][0] * s4 - self[2][3] * s0;
            adjugate[3][0] = self[1][1] * c1 - self[1][0] * c3 - self[1][2] * c0;
            adjugate[3][1] = self[0][0] * c3 - self[0][1] * c1 + self[0][2] * c0;
            adjugate[3][2] = self[3][1] * s1 - self[3][0] * s3 - self[3][2] * s0;
            adjugate[3][3] = self[2][0] * s3 - self[2][1] * s1 + self[2][2] * s0;
            adjugate / (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0)
        } else {
            let (tensor_l_inverse, tensor_u_inverse) = self.lu_decomposition_inverse();
            tensor_u_inverse * tensor_l_inverse
        }
    }
    /// Returns the inverse and determinant of the rank-2 tensor.
    pub fn inverse_and_determinant(&self) -> (TensorRank2<D, J, I>, TensorRank0) {
        if D == 2 {
            let mut adjugate = TensorRank2::<D, J, I>::zero();
            adjugate[0][0] = self[1][1];
            adjugate[0][1] = -self[0][1];
            adjugate[1][0] = -self[1][0];
            adjugate[1][1] = self[0][0];
            let determinant = self.determinant();
            (adjugate / determinant, determinant)
        } else if D == 3 {
            let mut adjugate = TensorRank2::<D, J, I>::zero();
            let c_00 = self[1][1] * self[2][2] - self[1][2] * self[2][1];
            let c_10 = self[1][2] * self[2][0] - self[1][0] * self[2][2];
            let c_20 = self[1][0] * self[2][1] - self[1][1] * self[2][0];
            let determinant = self[0][0] * c_00 + self[0][1] * c_10 + self[0][2] * c_20;
            adjugate[0][0] = c_00;
            adjugate[0][1] = self[0][2] * self[2][1] - self[0][1] * self[2][2];
            adjugate[0][2] = self[0][1] * self[1][2] - self[0][2] * self[1][1];
            adjugate[1][0] = c_10;
            adjugate[1][1] = self[0][0] * self[2][2] - self[0][2] * self[2][0];
            adjugate[1][2] = self[0][2] * self[1][0] - self[0][0] * self[1][2];
            adjugate[2][0] = c_20;
            adjugate[2][1] = self[0][1] * self[2][0] - self[0][0] * self[2][1];
            adjugate[2][2] = self[0][0] * self[1][1] - self[0][1] * self[1][0];
            (adjugate / determinant, determinant)
        } else if D == 4 {
            let mut adjugate = TensorRank2::<D, J, I>::zero();
            let s0 = self[0][0] * self[1][1] - self[0][1] * self[1][0];
            let s1 = self[0][0] * self[1][2] - self[0][2] * self[1][0];
            let s2 = self[0][0] * self[1][3] - self[0][3] * self[1][0];
            let s3 = self[0][1] * self[1][2] - self[0][2] * self[1][1];
            let s4 = self[0][1] * self[1][3] - self[0][3] * self[1][1];
            let s5 = self[0][2] * self[1][3] - self[0][3] * self[1][2];
            let c5 = self[2][2] * self[3][3] - self[2][3] * self[3][2];
            let c4 = self[2][1] * self[3][3] - self[2][3] * self[3][1];
            let c3 = self[2][1] * self[3][2] - self[2][2] * self[3][1];
            let c2 = self[2][0] * self[3][3] - self[2][3] * self[3][0];
            let c1 = self[2][0] * self[3][2] - self[2][2] * self[3][0];
            let c0 = self[2][0] * self[3][1] - self[2][1] * self[3][0];
            let determinant = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
            adjugate[0][0] = self[1][1] * c5 - self[1][2] * c4 + self[1][3] * c3;
            adjugate[0][1] = self[0][2] * c4 - self[0][1] * c5 - self[0][3] * c3;
            adjugate[0][2] = self[3][1] * s5 - self[3][2] * s4 + self[3][3] * s3;
            adjugate[0][3] = self[2][2] * s4 - self[2][1] * s5 - self[2][3] * s3;
            adjugate[1][0] = self[1][2] * c2 - self[1][0] * c5 - self[1][3] * c1;
            adjugate[1][1] = self[0][0] * c5 - self[0][2] * c2 + self[0][3] * c1;
            adjugate[1][2] = self[3][2] * s2 - self[3][0] * s5 - self[3][3] * s1;
            adjugate[1][3] = self[2][0] * s5 - self[2][2] * s2 + self[2][3] * s1;
            adjugate[2][0] = self[1][0] * c4 - self[1][1] * c2 + self[1][3] * c0;
            adjugate[2][1] = self[0][1] * c2 - self[0][0] * c4 - self[0][3] * c0;
            adjugate[2][2] = self[3][0] * s4 - self[3][1] * s2 + self[3][3] * s0;
            adjugate[2][3] = self[2][1] * s2 - self[2][0] * s4 - self[2][3] * s0;
            adjugate[3][0] = self[1][1] * c1 - self[1][0] * c3 - self[1][2] * c0;
            adjugate[3][1] = self[0][0] * c3 - self[0][1] * c1 + self[0][2] * c0;
            adjugate[3][2] = self[3][1] * s1 - self[3][0] * s3 - self[3][2] * s0;
            adjugate[3][3] = self[2][0] * s3 - self[2][1] * s1 + self[2][2] * s0;
            (adjugate / determinant, determinant)
        } else {
            panic!()
        }
    }
    /// Returns the inverse transpose of the rank-2 tensor.
    pub fn inverse_transpose(&self) -> Self {
        if D == 2 {
            let mut adjugate_transpose = TensorRank2::<D, I, J>::zero();
            adjugate_transpose[0][0] = self[1][1];
            adjugate_transpose[0][1] = -self[1][0];
            adjugate_transpose[1][0] = -self[0][1];
            adjugate_transpose[1][1] = self[0][0];
            adjugate_transpose / self.determinant()
        } else if D == 3 {
            let mut adjugate_transpose = TensorRank2::<D, I, J>::zero();
            let c_00 = self[1][1] * self[2][2] - self[1][2] * self[2][1];
            let c_10 = self[1][2] * self[2][0] - self[1][0] * self[2][2];
            let c_20 = self[1][0] * self[2][1] - self[1][1] * self[2][0];
            adjugate_transpose[0][0] = c_00;
            adjugate_transpose[1][0] = self[0][2] * self[2][1] - self[0][1] * self[2][2];
            adjugate_transpose[2][0] = self[0][1] * self[1][2] - self[0][2] * self[1][1];
            adjugate_transpose[0][1] = c_10;
            adjugate_transpose[1][1] = self[0][0] * self[2][2] - self[0][2] * self[2][0];
            adjugate_transpose[2][1] = self[0][2] * self[1][0] - self[0][0] * self[1][2];
            adjugate_transpose[0][2] = c_20;
            adjugate_transpose[1][2] = self[0][1] * self[2][0] - self[0][0] * self[2][1];
            adjugate_transpose[2][2] = self[0][0] * self[1][1] - self[0][1] * self[1][0];
            adjugate_transpose / (self[0][0] * c_00 + self[0][1] * c_10 + self[0][2] * c_20)
        } else if D == 4 {
            let mut adjugate_transpose = TensorRank2::<D, I, J>::zero();
            let s0 = self[0][0] * self[1][1] - self[0][1] * self[1][0];
            let s1 = self[0][0] * self[1][2] - self[0][2] * self[1][0];
            let s2 = self[0][0] * self[1][3] - self[0][3] * self[1][0];
            let s3 = self[0][1] * self[1][2] - self[0][2] * self[1][1];
            let s4 = self[0][1] * self[1][3] - self[0][3] * self[1][1];
            let s5 = self[0][2] * self[1][3] - self[0][3] * self[1][2];
            let c5 = self[2][2] * self[3][3] - self[2][3] * self[3][2];
            let c4 = self[2][1] * self[3][3] - self[2][3] * self[3][1];
            let c3 = self[2][1] * self[3][2] - self[2][2] * self[3][1];
            let c2 = self[2][0] * self[3][3] - self[2][3] * self[3][0];
            let c1 = self[2][0] * self[3][2] - self[2][2] * self[3][0];
            let c0 = self[2][0] * self[3][1] - self[2][1] * self[3][0];
            adjugate_transpose[0][0] = self[1][1] * c5 - self[1][2] * c4 + self[1][3] * c3;
            adjugate_transpose[1][0] = self[0][2] * c4 - self[0][1] * c5 - self[0][3] * c3;
            adjugate_transpose[2][0] = self[3][1] * s5 - self[3][2] * s4 + self[3][3] * s3;
            adjugate_transpose[3][0] = self[2][2] * s4 - self[2][1] * s5 - self[2][3] * s3;
            adjugate_transpose[0][1] = self[1][2] * c2 - self[1][0] * c5 - self[1][3] * c1;
            adjugate_transpose[1][1] = self[0][0] * c5 - self[0][2] * c2 + self[0][3] * c1;
            adjugate_transpose[2][1] = self[3][2] * s2 - self[3][0] * s5 - self[3][3] * s1;
            adjugate_transpose[3][1] = self[2][0] * s5 - self[2][2] * s2 + self[2][3] * s1;
            adjugate_transpose[0][2] = self[1][0] * c4 - self[1][1] * c2 + self[1][3] * c0;
            adjugate_transpose[1][2] = self[0][1] * c2 - self[0][0] * c4 - self[0][3] * c0;
            adjugate_transpose[2][2] = self[3][0] * s4 - self[3][1] * s2 + self[3][3] * s0;
            adjugate_transpose[3][2] = self[2][1] * s2 - self[2][0] * s4 - self[2][3] * s0;
            adjugate_transpose[0][3] = self[1][1] * c1 - self[1][0] * c3 - self[1][2] * c0;
            adjugate_transpose[1][3] = self[0][0] * c3 - self[0][1] * c1 + self[0][2] * c0;
            adjugate_transpose[2][3] = self[3][1] * s1 - self[3][0] * s3 - self[3][2] * s0;
            adjugate_transpose[3][3] = self[2][0] * s3 - self[2][1] * s1 + self[2][2] * s0;
            adjugate_transpose / (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0)
        } else {
            self.inverse().transpose()
        }
    }
    /// Returns the inverse transpose and determinant of the rank-2 tensor.
    pub fn inverse_transpose_and_determinant(&self) -> (Self, TensorRank0) {
        if D == 2 {
            let mut adjugate_transpose = TensorRank2::<D, I, J>::zero();
            adjugate_transpose[0][0] = self[1][1];
            adjugate_transpose[0][1] = -self[1][0];
            adjugate_transpose[1][0] = -self[0][1];
            adjugate_transpose[1][1] = self[0][0];
            let determinant = self.determinant();
            (adjugate_transpose / determinant, determinant)
        } else if D == 3 {
            let mut adjugate_transpose = TensorRank2::<D, I, J>::zero();
            let c_00 = self[1][1] * self[2][2] - self[1][2] * self[2][1];
            let c_10 = self[1][2] * self[2][0] - self[1][0] * self[2][2];
            let c_20 = self[1][0] * self[2][1] - self[1][1] * self[2][0];
            let determinant = self[0][0] * c_00 + self[0][1] * c_10 + self[0][2] * c_20;
            adjugate_transpose[0][0] = c_00;
            adjugate_transpose[1][0] = self[0][2] * self[2][1] - self[0][1] * self[2][2];
            adjugate_transpose[2][0] = self[0][1] * self[1][2] - self[0][2] * self[1][1];
            adjugate_transpose[0][1] = c_10;
            adjugate_transpose[1][1] = self[0][0] * self[2][2] - self[0][2] * self[2][0];
            adjugate_transpose[2][1] = self[0][2] * self[1][0] - self[0][0] * self[1][2];
            adjugate_transpose[0][2] = c_20;
            adjugate_transpose[1][2] = self[0][1] * self[2][0] - self[0][0] * self[2][1];
            adjugate_transpose[2][2] = self[0][0] * self[1][1] - self[0][1] * self[1][0];
            (adjugate_transpose / determinant, determinant)
        } else if D == 4 {
            let mut adjugate_transpose = TensorRank2::<D, I, J>::zero();
            let s0 = self[0][0] * self[1][1] - self[0][1] * self[1][0];
            let s1 = self[0][0] * self[1][2] - self[0][2] * self[1][0];
            let s2 = self[0][0] * self[1][3] - self[0][3] * self[1][0];
            let s3 = self[0][1] * self[1][2] - self[0][2] * self[1][1];
            let s4 = self[0][1] * self[1][3] - self[0][3] * self[1][1];
            let s5 = self[0][2] * self[1][3] - self[0][3] * self[1][2];
            let c5 = self[2][2] * self[3][3] - self[2][3] * self[3][2];
            let c4 = self[2][1] * self[3][3] - self[2][3] * self[3][1];
            let c3 = self[2][1] * self[3][2] - self[2][2] * self[3][1];
            let c2 = self[2][0] * self[3][3] - self[2][3] * self[3][0];
            let c1 = self[2][0] * self[3][2] - self[2][2] * self[3][0];
            let c0 = self[2][0] * self[3][1] - self[2][1] * self[3][0];
            let determinant = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
            adjugate_transpose[0][0] = self[1][1] * c5 - self[1][2] * c4 + self[1][3] * c3;
            adjugate_transpose[1][0] = self[0][2] * c4 - self[0][1] * c5 - self[0][3] * c3;
            adjugate_transpose[2][0] = self[3][1] * s5 - self[3][2] * s4 + self[3][3] * s3;
            adjugate_transpose[3][0] = self[2][2] * s4 - self[2][1] * s5 - self[2][3] * s3;
            adjugate_transpose[0][1] = self[1][2] * c2 - self[1][0] * c5 - self[1][3] * c1;
            adjugate_transpose[1][1] = self[0][0] * c5 - self[0][2] * c2 + self[0][3] * c1;
            adjugate_transpose[2][1] = self[3][2] * s2 - self[3][0] * s5 - self[3][3] * s1;
            adjugate_transpose[3][1] = self[2][0] * s5 - self[2][2] * s2 + self[2][3] * s1;
            adjugate_transpose[0][2] = self[1][0] * c4 - self[1][1] * c2 + self[1][3] * c0;
            adjugate_transpose[1][2] = self[0][1] * c2 - self[0][0] * c4 - self[0][3] * c0;
            adjugate_transpose[2][2] = self[3][0] * s4 - self[3][1] * s2 + self[3][3] * s0;
            adjugate_transpose[3][2] = self[2][1] * s2 - self[2][0] * s4 - self[2][3] * s0;
            adjugate_transpose[0][3] = self[1][1] * c1 - self[1][0] * c3 - self[1][2] * c0;
            adjugate_transpose[1][3] = self[0][0] * c3 - self[0][1] * c1 + self[0][2] * c0;
            adjugate_transpose[2][3] = self[3][1] * s1 - self[3][0] * s3 - self[3][2] * s0;
            adjugate_transpose[3][3] = self[2][0] * s3 - self[2][1] * s1 + self[2][2] * s0;
            (adjugate_transpose / determinant, determinant)
        } else {
            panic!()
        }
    }
    /// Returns the LU decomposition of the rank-2 tensor.
    pub fn lu_decomposition(&self) -> (TensorRank2<D, I, 88>, TensorRank2<D, 88, J>) {
        let mut tensor_l = TensorRank2::zero();
        let mut tensor_u = TensorRank2::zero();
        for i in 0..D {
            for k in i..D {
                tensor_u[i][k] = self[i][k];
                for j in 0..i {
                    tensor_u[i][k] -= tensor_l[i][j] * tensor_u[j][k];
                }
            }
            if tensor_u[i][i].abs() <= ABS_TOL {
                panic!("LU decomposition failed (zero pivot).")
            }
            for k in i..D {
                if i == k {
                    tensor_l[i][k] = 1.0
                } else {
                    tensor_l[k][i] = self[k][i];
                    for j in 0..i {
                        tensor_l[k][i] -= tensor_l[k][j] * tensor_u[j][i];
                    }
                    tensor_l[k][i] /= tensor_u[i][i]
                }
            }
        }
        (tensor_l, tensor_u)
    }
    /// Returns the inverse of the LU decomposition of the rank-2 tensor.
    pub fn lu_decomposition_inverse(&self) -> (TensorRank2<D, 88, I>, TensorRank2<D, J, 88>) {
        let mut tensor_l = TensorRank2::zero();
        let mut tensor_u = TensorRank2::zero();
        for i in 0..D {
            for k in i..D {
                tensor_u[i][k] = self[i][k];
                for j in 0..i {
                    tensor_u[i][k] -= tensor_l[i][j] * tensor_u[j][k];
                }
            }
            if tensor_u[i][i].abs() <= ABS_TOL {
                panic!("LU decomposition failed (zero pivot).")
            }
            for k in i..D {
                if i == k {
                    tensor_l[i][k] = 1.0
                } else {
                    tensor_l[k][i] = self[k][i];
                    for j in 0..i {
                        tensor_l[k][i] -= tensor_l[k][j] * tensor_u[j][i];
                    }
                    tensor_l[k][i] /= tensor_u[i][i]
                }
            }
        }
        //
        // above is copied from lu_decomposition
        //
        let mut sum;
        for i in 0..D {
            tensor_l[i][i] = 1.0 / tensor_l[i][i];
            for j in 0..i {
                sum = 0.0;
                for k in j..i {
                    sum += tensor_l[i][k] * tensor_l[k][j];
                }
                tensor_l[i][j] = -sum * tensor_l[i][i];
            }
        }
        for i in 0..D {
            tensor_u[i][i] = 1.0 / tensor_u[i][i];
            for j in 0..i {
                sum = 0.0;
                for k in j..i {
                    sum += tensor_u[j][k] * tensor_u[k][i];
                }
                tensor_u[j][i] = -sum * tensor_u[i][i];
            }
        }
        (tensor_l, tensor_u)
    }
}

impl<const I: usize> TensorRank2<3, I, I> {
    /// Returns the matrix logarithm of the 3x3 symmetric tensor.
    pub fn logm(&self) -> Self {
        if self.is_diagonal() {
            let mut logm = TensorRank2::zero();
            logm.iter_mut()
                .enumerate()
                .zip(self.iter())
                .for_each(|((i, logm_i), self_i)| logm_i[i] = self_i[i].ln());
            logm
        } else {
            let tensor = self - &TensorRank2::identity();
            let norm = tensor.norm();
            if norm < 1e-2 {
                let num_terms = if norm < 1e-4 {
                    2
                } else if norm < 1e-3 {
                    3
                } else {
                    5
                };
                let mut logm = tensor.clone();
                let mut power = tensor.clone();
                (2..=num_terms).for_each(|k| {
                    power *= &tensor;
                    logm += &power / (if k % 2 == 0 { -1.0 } else { 1.0 } / k as f64);
                });
                logm
            } else if self.is_symmetric() {
                let mut eigenvalues = solve_cubic_symmetric(self.invariants());
                if eigenvalues.iter().any(|eigenvalue| eigenvalue <= &0.0) {
                    panic!("Symmetric matrix has a non-positive eigenvalue")
                }
                let eigenvectors = find_orthonormal_eigenvectors(&eigenvalues, self);
                eigenvalues
                    .iter_mut()
                    .for_each(|eigenvalue| *eigenvalue = eigenvalue.ln());
                reconstruct_symmetric(eigenvalues, eigenvectors)
            } else {
                panic!("Matrix logarithm only implemented for symmetric cases")
            }
        }
    }
    /// Returns the derivative of the matrix logarithm of the 3x3 symmetric tensor.
    pub fn dlogm(&self) -> TensorRank4<3, I, I, I, I> {
        if self.is_diagonal() {
            let mut dlogm = TensorRank4::zero();
            dlogm.iter_mut().enumerate().for_each(|(i, dlogm_i)| {
                dlogm_i.iter_mut().enumerate().for_each(|(j, dlogm_ij)| {
                    dlogm_ij.iter_mut().enumerate().for_each(|(k, dlogm_ijk)| {
                        dlogm_ijk
                            .iter_mut()
                            .enumerate()
                            .filter(|(l, _)| i == k && &j == l)
                            .for_each(|(_, dlogm_ijkl)| {
                                *dlogm_ijkl = if assert_eq_within_tols(&self[i][i], &self[j][j])
                                    .is_ok()
                                {
                                    1.0 / self[j][j]
                                } else {
                                    (self[i][i].ln() - self[j][j].ln()) / (self[i][i] - self[j][j])
                                }
                            })
                    })
                })
            });
            dlogm
        } else if self.is_symmetric() {
            let eigenvalues = solve_cubic_symmetric(self.invariants());
            if eigenvalues.iter().any(|eigenvalue| eigenvalue <= &0.0) {
                panic!("Symmetric matrix has a non-positive eigenvalue")
            }
            let divided_difference: Self = eigenvalues
                .iter()
                .map(|eigenvalue_i| {
                    eigenvalues
                        .iter()
                        .map(|eigenvalue_j| {
                            if assert_eq_within_tols(eigenvalue_i, eigenvalue_j).is_ok() {
                                1.0 / eigenvalue_j
                            } else {
                                (eigenvalue_i.ln() - eigenvalue_j.ln())
                                    / (eigenvalue_i - eigenvalue_j)
                            }
                        })
                        .collect()
                })
                .collect();
            let eigenvectors = find_orthonormal_eigenvectors(&eigenvalues, self).transpose();
            eigenvectors.iter().map(|eigenvector_i|
                eigenvectors.iter().map(|eigenvector_j|
                    eigenvectors.iter().map(|eigenvector_k|
                        eigenvectors.iter().map(|eigenvector_l|
                            eigenvector_i.iter().zip(eigenvector_k.iter().zip(divided_difference.iter())).map(|(eigenvector_ip, (eigenvector_kp, divided_difference_p))|
                                eigenvector_j.iter().zip(eigenvector_l.iter().zip(divided_difference_p.iter())).map(|(eigenvector_jq, (eigenvector_lq, divided_difference_pq))|
                                    eigenvector_ip * eigenvector_kp * divided_difference_pq * eigenvector_jq * eigenvector_lq
                                ).sum::<TensorRank0>()
                            ).sum()
                        ).collect()
                    ).collect()
                ).collect()
            ).collect()
        } else {
            panic!("Matrix logarithm only implemented for symmetric cases")
        }
    }
    /// Returns the invariants of the 3x3 symmetric tensor.
    pub fn invariants(&self) -> TensorRank0List<3> {
        let trace = self.trace();
        TensorRank0List::new([
            trace,
            0.5 * (trace.powi(2) - self.squared_trace()),
            self.determinant(),
        ])
    }
}

fn solve_cubic_symmetric(coefficients: TensorRank0List<3>) -> TensorRank0List<3> {
    let c2 = coefficients[0];
    let c1 = coefficients[1];
    let c0 = coefficients[2];
    let p = c1 - c2 * c2 / 3.0;
    let q = -(2.0 * c2.powi(3) - 9.0 * c2 * c1 + 27.0 * c0) / 27.0;
    if p.abs() < ABS_TOL {
        let t = (-q).cbrt();
        let lambda = t + c2 / 3.0;
        return TensorRank0List::new([lambda; _]);
    }
    let discriminant = -4.0 * p * p * p - 27.0 * q * q;
    if discriminant >= ABS_TOL {
        let sqrt_term = (-p / 3.0).sqrt();
        let cos_arg = 3.0 * q / (2.0 * p * (-p / 3.0).sqrt());
        let cos_arg = cos_arg.clamp(-1.0, 1.0);
        let theta = cos_arg.acos();
        let mut lambdas = [
            2.0 * sqrt_term * (theta / 3.0).cos() + c2 / 3.0,
            2.0 * sqrt_term * ((theta + TAU) / 3.0).cos() + c2 / 3.0,
            2.0 * sqrt_term * ((theta + 2.0 * TAU) / 3.0).cos() + c2 / 3.0,
        ];
        lambdas.sort_by(|a, b| b.partial_cmp(a).unwrap());
        TensorRank0List::new(lambdas)
    } else {
        panic!("Symmetric matrix produced complex eigenvalues");
    }
}

fn find_orthonormal_eigenvectors<const I: usize>(
    eigenvalues: &TensorRank0List<3>,
    tensor: &TensorRank2<3, I, I>,
) -> TensorRank2<3, I, I> {
    let mut eigenvectors = eigenvalues
        .iter()
        .map(|eigenvalue| eigenvector_symmetric(eigenvalue, tensor))
        .collect::<TensorRank2<3, I, I>>();
    eigenvectors[0].normalize();
    let proj1 = &eigenvectors[1] * &eigenvectors[0];
    for i in 0..3 {
        eigenvectors[1][i] -= proj1 * eigenvectors[0][i];
    }
    eigenvectors[1].normalize();
    eigenvectors[2] = eigenvectors[0].cross(&eigenvectors[1]);
    eigenvectors
}

fn eigenvector_symmetric<const I: usize>(
    eigenvalue: &TensorRank0,
    tensor: &TensorRank2<3, I, I>,
) -> TensorRank1<3, I> {
    let m = tensor - TensorRank2::identity() * eigenvalue;
    let mut pivot_row = 0;
    m.iter().enumerate().for_each(|(i, m_i)| {
        if m_i[i].abs() < m[pivot_row][pivot_row].abs() {
            pivot_row = i;
        }
    });
    if pivot_row == 0 {
        m[1].cross(&m[2])
    } else if pivot_row == 1 {
        m[0].cross(&m[2])
    } else {
        m[0].cross(&m[1])
    }
    .normalized()
}

fn reconstruct_symmetric<const I: usize>(
    eigenvalues: TensorRank0List<3>,
    eigenvectors: TensorRank2<3, I, I>,
) -> TensorRank2<3, I, I> {
    let mut tensor = TensorRank2::zero();
    eigenvalues
        .iter()
        .zip(eigenvectors.iter())
        .for_each(|(eigenvalue, eigenvector)| {
            tensor
                .iter_mut()
                .zip(eigenvector.iter())
                .for_each(|(tensor_i, eigenvector_i)| {
                    tensor_i.iter_mut().zip(eigenvector.iter()).for_each(
                        |(tensor_ij, eigenvector_j)| {
                            *tensor_ij += eigenvalue * eigenvector_i * eigenvector_j
                        },
                    )
                })
        });
    tensor
}

impl<const D: usize, const I: usize, const J: usize> Hessian for TensorRank2<D, I, J> {
    fn fill_into(self, square_matrix: &mut SquareMatrix) {
        self.into_iter().enumerate().for_each(|(i, self_i)| {
            self_i
                .into_iter()
                .enumerate()
                .for_each(|(j, self_ij)| square_matrix[i][j] = self_ij)
        })
    }
}

impl<const D: usize, const I: usize, const J: usize> Rank2 for TensorRank2<D, I, J> {
    type Transpose = TensorRank2<D, J, I>;
    fn deviatoric(&self) -> Self {
        Self::identity() * (self.trace() / -(D as TensorRank0)) + self
    }
    fn deviatoric_and_trace(&self) -> (Self, TensorRank0) {
        let trace = self.trace();
        (
            Self::identity() * (trace / -(D as TensorRank0)) + self,
            trace,
        )
    }
    fn is_diagonal(&self) -> bool {
        self.iter()
            .enumerate()
            .map(|(i, self_i)| {
                self_i
                    .iter()
                    .enumerate()
                    .map(|(j, self_ij)| (self_ij.abs() < ABS_TOL) as u8 * (i != j) as u8)
                    .sum::<u8>()
            })
            .sum::<u8>()
            == (D.pow(2) - D) as u8
    }
    fn is_identity(&self) -> bool {
        self.iter().enumerate().all(|(i, self_i)| {
            self_i
                .iter()
                .enumerate()
                .all(|(j, self_ij)| self_ij == &((i == j) as u8 as TensorRank0))
        })
    }
    fn is_symmetric(&self) -> bool {
        self.iter().enumerate().all(|(i, self_i)| {
            self_i
                .iter()
                .zip(self.iter())
                .all(|(self_ij, self_j)| self_ij == &self_j[i])
        })
    }
    fn squared_trace(&self) -> TensorRank0 {
        self.iter()
            .enumerate()
            .map(|(i, self_i)| {
                self_i
                    .iter()
                    .zip(self.iter())
                    .map(|(self_ij, self_j)| self_ij * self_j[i])
                    .sum::<TensorRank0>()
            })
            .sum()
    }
    fn trace(&self) -> TensorRank0 {
        self.iter().enumerate().map(|(i, self_i)| self_i[i]).sum()
    }
    fn transpose(&self) -> Self::Transpose {
        (0..D)
            .map(|i| (0..D).map(|j| self[j][i]).collect())
            // .map(|i| self.iter().map(|self_j| self_j[i]).collect())
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Tensor for TensorRank2<D, I, J> {
    type Item = TensorRank1<D, J>;
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut()
    }
    fn norm_inf(&self) -> TensorRank0 {
        self.iter()
            .map(|tensor_rank_1| {
                tensor_rank_1
                    .iter()
                    .fold(0.0, |acc, entry| entry.abs().max(acc))
            })
            .reduce(TensorRank0::max)
            .unwrap()
    }
    fn num_entries(&self) -> usize {
        D * D
    }
}

impl<const D: usize, const I: usize, const J: usize> IntoIterator for TensorRank2<D, I, J> {
    type Item = TensorRank1<D, J>;
    type IntoIter = std::array::IntoIter<Self::Item, D>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize, const I: usize, const J: usize> TensorArray for TensorRank2<D, I, J> {
    type Array = [[TensorRank0; D]; D];
    type Item = TensorRank1<D, J>;
    fn as_array(&self) -> Self::Array {
        let mut array = [[0.0; D]; D];
        array
            .iter_mut()
            .zip(self.iter())
            .for_each(|(entry, tensor_rank_1)| *entry = tensor_rank_1.as_array());
        array
    }
    fn identity() -> Self {
        (0..D)
            .map(|i| (0..D).map(|j| ((i == j) as u8) as TensorRank0).collect())
            .collect()
    }
    fn new(array: Self::Array) -> Self {
        array.into_iter().map(Self::Item::new).collect()
    }
    fn zero() -> Self {
        Self(from_fn(|_| Self::Item::zero()))
    }
}

impl<const D: usize, const I: usize, const J: usize> Solution for TensorRank2<D, I, J> {
    fn decrement_from(&mut self, other: &Vector) {
        self.iter_mut()
            .flat_map(|x| x.iter_mut())
            .zip(other.iter())
            .for_each(|(self_i, vector_i)| *self_i -= vector_i)
    }
    fn decrement_from_chained(&mut self, other: &mut Vector, vector: Vector) {
        self.iter_mut()
            .flat_map(|x| x.iter_mut())
            .chain(other.iter_mut())
            .zip(vector)
            .for_each(|(entry_i, vector_i)| *entry_i -= vector_i)
    }
}

impl<const D: usize, const I: usize, const J: usize> Jacobian for TensorRank2<D, I, J> {
    fn fill_into(self, vector: &mut Vector) {
        self.into_iter()
            .flatten()
            .zip(vector.iter_mut())
            .for_each(|(self_i, vector_i)| *vector_i = self_i)
    }
    fn fill_into_chained(self, other: Vector, vector: &mut Vector) {
        self.into_iter()
            .flatten()
            .chain(other)
            .zip(vector.iter_mut())
            .for_each(|(self_i, vector_i)| *vector_i = self_i)
    }
}

impl<const D: usize, const I: usize, const J: usize> Sub<Vector> for TensorRank2<D, I, J> {
    type Output = Self;
    fn sub(mut self, vector: Vector) -> Self::Output {
        self.iter_mut().enumerate().for_each(|(i, self_i)| {
            self_i
                .iter_mut()
                .enumerate()
                .for_each(|(j, self_ij)| *self_ij -= vector[D * i + j])
        });
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Sub<&Vector> for TensorRank2<D, I, J> {
    type Output = Self;
    fn sub(mut self, vector: &Vector) -> Self::Output {
        self.iter_mut().enumerate().for_each(|(i, self_i)| {
            self_i
                .iter_mut()
                .enumerate()
                .for_each(|(j, self_ij)| *self_ij -= vector[D * i + j])
        });
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    From<TensorRank4<D, I, J, K, L>> for TensorRank2<9, 88, 99>
{
    fn from(tensor_rank_4: TensorRank4<D, I, J, K, L>) -> Self {
        assert_eq!(D, 3);
        tensor_rank_4
            .into_iter()
            .flatten()
            .map(|entry_ij| entry_ij.into_iter().flatten().collect())
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    From<&TensorRank4<D, I, J, K, L>> for TensorRank2<9, 88, 99>
{
    fn from(tensor_rank_4: &TensorRank4<D, I, J, K, L>) -> Self {
        assert_eq!(D, 3);
        tensor_rank_4
            .clone()
            .into_iter()
            .flatten()
            .map(|entry_ij| entry_ij.into_iter().flatten().collect())
            .collect()
    }
}

impl<const I: usize> From<TensorRank2<3, I, 1>> for TensorRank2<3, I, 0> {
    fn from(tensor_rank_2: TensorRank2<3, I, 1>) -> Self {
        unsafe { transmute::<TensorRank2<3, I, 1>, TensorRank2<3, I, 0>>(tensor_rank_2) }
    }
}

impl<const I: usize> From<TensorRank2<3, I, 2>> for TensorRank2<3, I, 0> {
    fn from(tensor_rank_2: TensorRank2<3, I, 2>) -> Self {
        unsafe { transmute::<TensorRank2<3, I, 2>, TensorRank2<3, I, 0>>(tensor_rank_2) }
    }
}

impl<const J: usize> From<TensorRank2<3, 0, J>> for TensorRank2<3, 1, J> {
    fn from(tensor_rank_2: TensorRank2<3, 0, J>) -> Self {
        unsafe { transmute::<TensorRank2<3, 0, J>, TensorRank2<3, 1, J>>(tensor_rank_2) }
    }
}

impl<const J: usize> From<TensorRank2<3, 1, J>> for TensorRank2<3, 0, J> {
    fn from(tensor_rank_2: TensorRank2<3, 1, J>) -> Self {
        unsafe { transmute::<TensorRank2<3, 1, J>, TensorRank2<3, 0, J>>(tensor_rank_2) }
    }
}

impl From<TensorRank2<3, 0, 0>> for TensorRank2<3, 1, 1> {
    fn from(tensor_rank_2: TensorRank2<3, 0, 0>) -> Self {
        unsafe { transmute::<TensorRank2<3, 0, 0>, TensorRank2<3, 1, 1>>(tensor_rank_2) }
    }
}

impl<const D: usize, const I: usize, const J: usize> From<Vector> for TensorRank2<D, I, J> {
    fn from(_vector: Vector) -> Self {
        unimplemented!()
    }
}

impl<const D: usize, const I: usize, const J: usize> FromIterator<TensorRank1<D, J>>
    for TensorRank2<D, I, J>
{
    fn from_iter<Ii: IntoIterator<Item = TensorRank1<D, J>>>(into_iterator: Ii) -> Self {
        let mut tensor_rank_2 = Self::zero();
        tensor_rank_2
            .iter_mut()
            .zip(into_iterator)
            .for_each(|(tensor_rank_2_i, value_i)| *tensor_rank_2_i = value_i);
        tensor_rank_2
    }
}

impl<const D: usize, const I: usize, const J: usize> Index<usize> for TensorRank2<D, I, J> {
    type Output = TensorRank1<D, J>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize, const I: usize, const J: usize> IndexMut<usize> for TensorRank2<D, I, J> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const D: usize, const I: usize, const J: usize> std::iter::Sum for TensorRank2<D, I, J> {
    fn sum<Ii>(iter: Ii) -> Self
    where
        Ii: Iterator<Item = Self>,
    {
        let mut output = Self::zero();
        iter.for_each(|item| output += item);
        output
    }
}

impl<const D: usize, const I: usize, const J: usize> Div<TensorRank0> for TensorRank2<D, I, J> {
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= &tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Div<TensorRank0> for &TensorRank2<D, I, J> {
    type Output = TensorRank2<D, I, J>;
    fn div(self, tensor_rank_0: TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i / tensor_rank_0).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Div<&TensorRank0> for TensorRank2<D, I, J> {
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Div<&TensorRank0> for &TensorRank2<D, I, J> {
    type Output = TensorRank2<D, I, J>;
    fn div(self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i / tensor_rank_0).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> DivAssign<TensorRank0>
    for TensorRank2<D, I, J>
{
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i /= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize> DivAssign<&TensorRank0>
    for TensorRank2<D, I, J>
{
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i /= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<TensorRank0> for TensorRank2<D, I, J> {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= &tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<TensorRank0> for &TensorRank2<D, I, J> {
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_0: TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_0).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<&TensorRank0> for TensorRank2<D, I, J> {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<&TensorRank0> for &TensorRank2<D, I, J> {
    type Output = TensorRank2<D, I, J>;
    fn mul(self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_0).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> MulAssign<TensorRank0>
    for TensorRank2<D, I, J>
{
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i *= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize> MulAssign<&TensorRank0>
    for TensorRank2<D, I, J>
{
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i *= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<TensorRank1<D, J>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank1<D, I>;
    fn mul(self, tensor_rank_1: TensorRank1<D, J>) -> Self::Output {
        self.iter().map(|self_i| self_i * &tensor_rank_1).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<&TensorRank1<D, J>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank1<D, I>;
    fn mul(self, tensor_rank_1: &TensorRank1<D, J>) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_1).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<TensorRank1<D, J>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank1<D, I>;
    fn mul(self, tensor_rank_1: TensorRank1<D, J>) -> Self::Output {
        self.iter().map(|self_i| self_i * &tensor_rank_1).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<&TensorRank1<D, J>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank1<D, I>;
    fn mul(self, tensor_rank_1: &TensorRank1<D, J>) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_1).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Add for TensorRank2<D, I, J> {
    type Output = Self;
    fn add(mut self, tensor_rank_2: Self) -> Self::Output {
        self += tensor_rank_2;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Add<&Self> for TensorRank2<D, I, J> {
    type Output = Self;
    fn add(mut self, tensor_rank_2: &Self) -> Self::Output {
        self += tensor_rank_2;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Add<TensorRank2<D, I, J>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank2<D, I, J>;
    fn add(self, mut tensor_rank_2: TensorRank2<D, I, J>) -> Self::Output {
        tensor_rank_2 += self;
        tensor_rank_2
    }
}

impl<const D: usize, const I: usize, const J: usize> AddAssign for TensorRank2<D, I, J> {
    fn add_assign(&mut self, tensor_rank_2: Self) {
        self.iter_mut()
            .zip(tensor_rank_2.iter())
            .for_each(|(self_i, tensor_rank_2_i)| *self_i += tensor_rank_2_i);
    }
}

impl<const D: usize, const I: usize, const J: usize> AddAssign<&Self> for TensorRank2<D, I, J> {
    fn add_assign(&mut self, tensor_rank_2: &Self) {
        self.iter_mut()
            .zip(tensor_rank_2.iter())
            .for_each(|(self_i, tensor_rank_2_i)| *self_i += tensor_rank_2_i);
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize> Mul<TensorRank2<D, J, K>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank2<D, I, K>;
    fn mul(self, tensor_rank_2: TensorRank2<D, J, K>) -> Self::Output {
        self.iter()
            .map(|self_i| {
                self_i
                    .iter()
                    .zip(tensor_rank_2.iter())
                    .map(|(self_ij, tensor_rank_2_j)| tensor_rank_2_j * self_ij)
                    .sum()
            })
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize> Mul<&TensorRank2<D, J, K>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank2<D, I, K>;
    fn mul(self, tensor_rank_2: &TensorRank2<D, J, K>) -> Self::Output {
        self.iter()
            .map(|self_i| {
                self_i
                    .iter()
                    .zip(tensor_rank_2.iter())
                    .map(|(self_ij, tensor_rank_2_j)| tensor_rank_2_j * self_ij)
                    .sum()
            })
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize> Mul<TensorRank2<D, J, K>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank2<D, I, K>;
    fn mul(self, tensor_rank_2: TensorRank2<D, J, K>) -> Self::Output {
        self.iter()
            .map(|self_i| {
                self_i
                    .iter()
                    .zip(tensor_rank_2.iter())
                    .map(|(self_ij, tensor_rank_2_j)| tensor_rank_2_j * self_ij)
                    .sum()
            })
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize> Mul<&TensorRank2<D, J, K>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank2<D, I, K>;
    fn mul(self, tensor_rank_2: &TensorRank2<D, J, K>) -> Self::Output {
        self.iter()
            .map(|self_i| {
                self_i
                    .iter()
                    .zip(tensor_rank_2.iter())
                    .map(|(self_ij, tensor_rank_2_j)| tensor_rank_2_j * self_ij)
                    .sum()
            })
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> MulAssign<TensorRank2<D, J, J>>
    for TensorRank2<D, I, J>
{
    fn mul_assign(&mut self, tensor_rank_2: TensorRank2<D, J, J>) {
        *self = &*self * tensor_rank_2
    }
}

impl<const D: usize, const I: usize, const J: usize> MulAssign<&TensorRank2<D, J, J>>
    for TensorRank2<D, I, J>
{
    fn mul_assign(&mut self, tensor_rank_2: &TensorRank2<D, J, J>) {
        *self = &*self * tensor_rank_2
    }
}

impl<const D: usize, const I: usize, const J: usize> Sub for TensorRank2<D, I, J> {
    type Output = Self;
    fn sub(mut self, tensor_rank_2: Self) -> Self::Output {
        self -= tensor_rank_2;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Sub<&Self> for TensorRank2<D, I, J> {
    type Output = Self;
    fn sub(mut self, tensor_rank_2: &Self) -> Self::Output {
        self -= tensor_rank_2;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Sub<TensorRank2<D, I, J>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank2<D, I, J>;
    fn sub(self, tensor_rank_2: TensorRank2<D, I, J>) -> Self::Output {
        let mut output = self.clone();
        output -= tensor_rank_2;
        output
    }
}

impl<const D: usize, const I: usize, const J: usize> Sub for &TensorRank2<D, I, J> {
    type Output = TensorRank2<D, I, J>;
    fn sub(self, tensor_rank_2: Self) -> Self::Output {
        let mut output = self.clone();
        output -= tensor_rank_2;
        output
    }
}

impl<const D: usize, const I: usize, const J: usize> SubAssign for TensorRank2<D, I, J> {
    fn sub_assign(&mut self, tensor_rank_2: Self) {
        self.iter_mut()
            .zip(tensor_rank_2.iter())
            .for_each(|(self_i, tensor_rank_2_i)| *self_i -= tensor_rank_2_i);
    }
}

impl<const D: usize, const I: usize, const J: usize> SubAssign<&Self> for TensorRank2<D, I, J> {
    fn sub_assign(&mut self, tensor_rank_2: &Self) {
        self.iter_mut()
            .zip(tensor_rank_2.iter())
            .for_each(|(self_i, tensor_rank_2_i)| *self_i -= tensor_rank_2_i);
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<TensorRank1List<D, J, W>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank1List<D, I, W>;
    fn mul(self, tensor_rank_1_list: TensorRank1List<D, J, W>) -> Self::Output {
        tensor_rank_1_list
            .iter()
            .map(|tensor_rank_1| &self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<&TensorRank1List<D, J, W>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank1List<D, I, W>;
    fn mul(self, tensor_rank_1_list: &TensorRank1List<D, J, W>) -> Self::Output {
        tensor_rank_1_list
            .iter()
            .map(|tensor_rank_1| &self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<TensorRank1List<D, J, W>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank1List<D, I, W>;
    fn mul(self, tensor_rank_1_list: TensorRank1List<D, J, W>) -> Self::Output {
        tensor_rank_1_list
            .iter()
            .map(|tensor_rank_1| self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<&TensorRank1List<D, J, W>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank1List<D, I, W>;
    fn mul(self, tensor_rank_1_list: &TensorRank1List<D, J, W>) -> Self::Output {
        tensor_rank_1_list
            .iter()
            .map(|tensor_rank_1| self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<TensorRank1Vec<D, J>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank1Vec<D, I>;
    fn mul(self, tensor_rank_1_vec: TensorRank1Vec<D, J>) -> Self::Output {
        tensor_rank_1_vec
            .iter()
            .map(|tensor_rank_1| &self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<&TensorRank1Vec<D, J>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank1Vec<D, I>;
    fn mul(self, tensor_rank_1_vec: &TensorRank1Vec<D, J>) -> Self::Output {
        tensor_rank_1_vec
            .iter()
            .map(|tensor_rank_1| &self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<TensorRank1Vec<D, J>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank1Vec<D, I>;
    fn mul(self, tensor_rank_1_vec: TensorRank1Vec<D, J>) -> Self::Output {
        tensor_rank_1_vec
            .iter()
            .map(|tensor_rank_1| self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<&TensorRank1Vec<D, J>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank1Vec<D, I>;
    fn mul(self, tensor_rank_1_vec: &TensorRank1Vec<D, J>) -> Self::Output {
        tensor_rank_1_vec
            .iter()
            .map(|tensor_rank_1| self * tensor_rank_1)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const W: usize, const X: usize>
    Mul<TensorRank2List2D<D, J, K, W, X>> for TensorRank2<D, I, J>
{
    type Output = TensorRank2List2D<D, I, K, W, X>;
    fn mul(self, tensor_rank_2_list_2d: TensorRank2List2D<D, J, K, W, X>) -> Self::Output {
        tensor_rank_2_list_2d
            .iter()
            .map(|tensor_rank_2_list_2d_entry| {
                tensor_rank_2_list_2d_entry
                    .iter()
                    .map(|tensor_rank_2| &self * tensor_rank_2)
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const W: usize, const X: usize>
    Mul<TensorRank2List2D<D, J, K, W, X>> for &TensorRank2<D, I, J>
{
    type Output = TensorRank2List2D<D, I, K, W, X>;
    fn mul(self, tensor_rank_2_list_2d: TensorRank2List2D<D, J, K, W, X>) -> Self::Output {
        tensor_rank_2_list_2d
            .iter()
            .map(|tensor_rank_2_list_2d_entry| {
                tensor_rank_2_list_2d_entry
                    .iter()
                    .map(|tensor_rank_2| self * tensor_rank_2)
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize> Mul<TensorRank2Vec2D<D, J, K>>
    for TensorRank2<D, I, J>
{
    type Output = TensorRank2Vec2D<D, I, K>;
    fn mul(self, tensor_rank_2_list_2d: TensorRank2Vec2D<D, J, K>) -> Self::Output {
        tensor_rank_2_list_2d
            .iter()
            .map(|tensor_rank_2_list_2d_entry| {
                tensor_rank_2_list_2d_entry
                    .iter()
                    .map(|tensor_rank_2| &self * tensor_rank_2)
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize> Mul<TensorRank2Vec2D<D, J, K>>
    for &TensorRank2<D, I, J>
{
    type Output = TensorRank2Vec2D<D, I, K>;
    fn mul(self, tensor_rank_2_list_2d: TensorRank2Vec2D<D, J, K>) -> Self::Output {
        tensor_rank_2_list_2d
            .iter()
            .map(|tensor_rank_2_list_2d_entry| {
                tensor_rank_2_list_2d_entry
                    .iter()
                    .map(|tensor_rank_2| self * tensor_rank_2)
                    .collect()
            })
            .collect()
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<const I: usize, const J: usize, const K: usize, const L: usize> Div<TensorRank4<3, I, J, K, L>>
    for &TensorRank2<3, I, J>
{
    type Output = TensorRank2<3, K, L>;
    fn div(self, tensor_rank_4: TensorRank4<3, I, J, K, L>) -> Self::Output {
        let tensor_rank_2: TensorRank2<9, 88, 99> = tensor_rank_4.into();
        let output_tensor_rank_1 = tensor_rank_2.inverse() * self.as_tensor_rank_1();
        let mut output = TensorRank2::zero();
        output.iter_mut().enumerate().for_each(|(i, output_i)| {
            output_i
                .iter_mut()
                .enumerate()
                .for_each(|(j, output_ij)| *output_ij = output_tensor_rank_1[3 * i + j])
        });
        output
    }
}
