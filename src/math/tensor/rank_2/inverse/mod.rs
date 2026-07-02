#[cfg(test)]
mod test;

use super::{Rank2, Tensor, TensorArray, TensorRank0, TensorRank2};
use crate::ABS_TOL;

impl<const D: usize, const I: usize, const J: usize> TensorRank2<D, I, J> {
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
            let (_, u, p) = self.lu_decomposition();
            let num_swaps = p.iter().enumerate().filter(|(i, p_i)| p_i != &i).count();
            u.into_iter()
                .enumerate()
                .map(|(i, u_i)| u_i[i])
                .product::<TensorRank0>()
                * if num_swaps % 2 == 0 { 1.0 } else { -1.0 }
        }
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
            let (l_inverse, u_inverse, p) = self.lu_decomposition_inverse();
            let mut q = [0; D];
            p.into_iter().enumerate().for_each(|(i, p_i)| q[p_i] = i);
            u_inverse
                .into_iter()
                .map(|u_inverse_i| {
                    q.iter()
                        .map(|&q_j| {
                            u_inverse_i
                                .iter()
                                .zip(l_inverse.iter())
                                .map(|(u_inverse_ik, l_inverse_k)| u_inverse_ik * l_inverse_k[q_j])
                                .sum()
                        })
                        .collect()
                })
                .collect()
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
            (self.inverse(), self.determinant())
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
            (self.inverse_transpose(), self.determinant())
        }
    }
    /// Returns the LU decomposition of the rank-2 tensor.
    pub fn lu_decomposition(&self) -> (TensorRank2<D, I, 88>, TensorRank2<D, 88, J>, Vec<usize>) {
        let n = D;
        let mut p: Vec<usize> = (0..n).collect();
        let mut factor;
        let mut lu = self.clone();
        let mut max_row;
        let mut max_val;
        let mut pivot;
        for i in 0..n {
            max_row = i;
            max_val = lu[max_row][i].abs();
            for k in i + 1..n {
                if lu[k][i].abs() > max_val {
                    max_row = k;
                    max_val = lu[max_row][i].abs();
                }
            }
            if max_row != i {
                lu.0.swap(i, max_row);
                p.swap(i, max_row);
            }
            pivot = lu[i][i];
            if pivot.abs() < ABS_TOL {
                panic!("LU decomposition failed (zero pivot).")
            }
            for j in i + 1..n {
                if lu[j][i] != 0.0 {
                    lu[j][i] /= pivot;
                    factor = lu[j][i];
                    for k in i + 1..n {
                        lu[j][k] -= factor * lu[i][k];
                    }
                }
            }
        }
        let mut l = TensorRank2::identity();
        for i in 0..D {
            for j in 0..i {
                l[i][j] = lu[i][j]
            }
        }
        let mut u = TensorRank2::zero();
        for i in 0..D {
            for j in i..D {
                u[i][j] = lu[i][j]
            }
        }
        (l, u, p)
    }
    /// Returns the inverse of the LU decomposition of the rank-2 tensor.
    pub fn lu_decomposition_inverse(
        &self,
    ) -> (TensorRank2<D, I, 88>, TensorRank2<D, 88, J>, Vec<usize>) {
        let (mut tensor_l, mut tensor_u, p) = self.lu_decomposition();
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
        (tensor_l, tensor_u, p)
    }
}
