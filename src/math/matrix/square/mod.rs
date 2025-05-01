#[cfg(test)]
mod test;

#[cfg(test)]
use crate::math::test::ErrorTensor;

use crate::{
    ABS_TOL,
    math::{
        Hessian, Rank2, Tensor, TensorRank0, TensorRank2Vec2D, TensorVec, Vector,
        tensor::TensorError, write_tensor_rank_0,
    },
};
use std::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use nalgebra::DMatrix as Temporary;

/// A square matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct SquareMatrix(Vec<Vector>);

impl SquareMatrix {
    /// Returns the inverse of the square matrix.
    pub fn inverse(&self) -> Self {
        let now = std::time::Instant::now();

        let n = self.len();
        let mut asdf = Temporary::<TensorRank0>::zeros(n, n);
        self.iter().enumerate().for_each(|(i, entry_i)| {
            entry_i
                .iter()
                .enumerate()
                .for_each(|(j, entry_ij)| asdf[(i, j)] = *entry_ij)
        });
        let fdsa = asdf.try_inverse().unwrap();
        let mut inverse = Self::zero(n);
        inverse.iter_mut().enumerate().for_each(|(i, entry_i)| {
            entry_i
                .iter_mut()
                .enumerate()
                .for_each(|(j, entry_ij)| *entry_ij = fdsa[(i, j)])
        });

        println!(
            "Running inverse() took {} microseconds.",
            now.elapsed().as_micros()
        );
        inverse
    }
    /// Returns the LU decomposition of the square matrix.
    pub fn lu_decomposition(&self) -> (Self, Self) {
        let n = self.len();
        let mut tensor_l = Self::zero(n);
        let mut tensor_u = Self::zero(n);
        for i in 0..n {
            for k in i..n {
                tensor_u[i][k] = self[i][k];
                for j in 0..i {
                    tensor_u[i][k] -= tensor_l[i][j] * tensor_u[j][k];
                }
            }
            if tensor_u[i][i].abs() <= ABS_TOL {
                panic!("LU decomposition failed (zero pivot).")
            }
            for k in i..n {
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
}

#[cfg(test)]
impl ErrorTensor for SquareMatrix {
    fn error(
        &self,
        comparator: &Self,
        tol_abs: &TensorRank0,
        tol_rel: &TensorRank0,
    ) -> Option<usize> {
        let error_count = self
            .iter()
            .zip(comparator.iter())
            .map(|(self_i, comparator_i)| {
                self_i
                    .iter()
                    .zip(comparator_i.iter())
                    .filter(|&(&self_ij, &comparator_ij)| {
                        &(self_ij - comparator_ij).abs() >= tol_abs
                            && &(self_ij / comparator_ij - 1.0).abs() >= tol_rel
                    })
                    .count()
            })
            .sum();
        if error_count > 0 {
            Some(error_count)
        } else {
            None
        }
    }
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

impl fmt::Display for SquareMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\x1B[s")?;
        write!(f, "[[")?;
        self.iter().enumerate().try_for_each(|(i, row)| {
            row.iter()
                .try_for_each(|entry| write_tensor_rank_0(f, entry))?;
            if i + 1 < self.len() {
                writeln!(f, "\x1B[2D],")?;
                write!(f, "\x1B[u")?;
                write!(f, "\x1B[{}B [", i + 1)?;
            }
            Ok(())
        })?;
        write!(f, "\x1B[2D]]")
    }
}

impl<const D: usize, const I: usize, const J: usize> From<TensorRank2Vec2D<D, I, J>>
    for SquareMatrix
{
    fn from(tensor_rank_2_vec_2d: TensorRank2Vec2D<D, I, J>) -> Self {
        let mut matrix = Self::zero(tensor_rank_2_vec_2d.len() * D);
        tensor_rank_2_vec_2d
            .iter()
            .enumerate()
            .for_each(|(a, entry_a)| {
                entry_a.iter().enumerate().for_each(|(b, entry_ab)| {
                    entry_ab.iter().enumerate().for_each(|(i, entry_ab_i)| {
                        entry_ab_i.iter().enumerate().for_each(|(j, entry_ab_ij)| {
                            matrix[D * a + i][D * b + j] = *entry_ab_ij
                        })
                    })
                })
            });
        matrix
    }
}

impl FromIterator<Vector> for SquareMatrix {
    fn from_iter<Ii: IntoIterator<Item = Vector>>(into_iterator: Ii) -> Self {
        Self(Vec::from_iter(into_iterator))
    }
}

impl Index<usize> for SquareMatrix {
    type Output = Vector;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for SquareMatrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl Hessian for SquareMatrix {
    fn fill_into(self, square_matrix: &mut SquareMatrix) {
        self.into_iter()
            .zip(square_matrix.iter_mut())
            .for_each(|(self_i, square_matrix_i)| {
                self_i
                    .into_iter()
                    .zip(square_matrix_i.iter_mut())
                    .for_each(|(self_ij, square_matrix_ij)| *square_matrix_ij = self_ij)
            });
    }
    fn is_positive_definite(&self) -> bool {
        self.cholesky_decomposition().is_ok()
    }
}

impl Rank2 for SquareMatrix {
    type Transpose = Self;
    fn cholesky_decomposition(&self) -> Result<SquareMatrix, TensorError> {
        let mut check = 0.0;
        let mut tensor_l = SquareMatrix::zero(self.len());
        self.iter().enumerate().try_for_each(|(j, self_j)| {
            check = self_j[j]
                - tensor_l[j]
                    .iter()
                    .take(j)
                    .map(|tensor_l_jk| tensor_l_jk.powi(2))
                    .sum::<TensorRank0>();
            if check < 0.0 {
                Err(TensorError::NotPositiveDefinite)
            } else {
                tensor_l[j][j] = check.sqrt();
                self.iter().enumerate().skip(j + 1).for_each(|(i, self_i)| {
                    check = tensor_l[i]
                        .iter()
                        .zip(tensor_l[j].iter())
                        .take(j)
                        .map(|(tensor_l_ik, tensor_l_jk)| tensor_l_ik * tensor_l_jk)
                        .sum();
                    tensor_l[i][j] = (self_i[j] - check) / tensor_l[j][j];
                });
                Ok(())
            }
        })?;
        Ok(tensor_l)
    }
    fn deviatoric(&self) -> Self {
        let len = self.len();
        let scale = -self.trace() / len as TensorRank0;
        (0..len)
            .map(|i| {
                (0..len)
                    .map(|j| ((i == j) as u8) as TensorRank0 * scale)
                    .collect()
            })
            .collect::<Self>()
            + self
    }
    fn deviatoric_and_trace(&self) -> (Self, TensorRank0) {
        let len = self.len();
        let trace = self.trace();
        let scale = -trace / len as TensorRank0;
        (
            (0..len)
                .map(|i| {
                    (0..len)
                        .map(|j| ((i == j) as u8) as TensorRank0 * scale)
                        .collect()
                })
                .collect::<Self>()
                + self,
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
                    .map(|(j, self_ij)| (self_ij == &0.0) as u8 * (i != j) as u8)
                    .sum::<u8>()
            })
            .sum::<u8>()
            == (self.len().pow(2) - self.len()) as u8
    }
    fn is_identity(&self) -> bool {
        self.iter()
            .enumerate()
            .map(|(i, self_i)| {
                self_i
                    .iter()
                    .enumerate()
                    .map(|(j, self_ij)| (self_ij == &((i == j) as u8 as f64)) as u8)
                    .sum::<u8>()
            })
            .sum::<u8>()
            == self.len().pow(2) as u8
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
        (0..self.len())
            .map(|i| (0..self.len()).map(|j| self[j][i]).collect())
            .collect()
    }
}

impl Tensor for SquareMatrix {
    type Item = Vector;
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut()
    }
}

impl IntoIterator for SquareMatrix {
    type Item = Vector;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl TensorVec for SquareMatrix {
    type Item = Vector;
    type Slice<'a> = &'a [&'a [TensorRank0]];
    fn append(&mut self, other: &mut Self) {
        self.0.append(&mut other.0)
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn new(slice: Self::Slice<'_>) -> Self {
        slice
            .iter()
            .map(|slice_entry| Self::Item::new(slice_entry))
            .collect()
    }
    fn push(&mut self, item: Self::Item) {
        self.0.push(item)
    }
    fn zero(len: usize) -> Self {
        (0..len).map(|_| Self::Item::zero(len)).collect()
    }
}

impl Div<TensorRank0> for SquareMatrix {
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= &tensor_rank_0;
        self
    }
}

impl Div<&TensorRank0> for SquareMatrix {
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl DivAssign<TensorRank0> for SquareMatrix {
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= &tensor_rank_0);
    }
}

impl DivAssign<&TensorRank0> for SquareMatrix {
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= tensor_rank_0);
    }
}

impl Mul<TensorRank0> for SquareMatrix {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= &tensor_rank_0;
        self
    }
}

impl Mul<&TensorRank0> for SquareMatrix {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl Mul<&TensorRank0> for &SquareMatrix {
    type Output = SquareMatrix;
    fn mul(self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_0).collect()
    }
}

impl MulAssign<TensorRank0> for SquareMatrix {
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= &tensor_rank_0);
    }
}

impl MulAssign<&TensorRank0> for SquareMatrix {
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= tensor_rank_0);
    }
}

impl Mul<Vector> for SquareMatrix {
    type Output = Vector;
    fn mul(self, vector: Vector) -> Self::Output {
        self.iter().map(|self_i| self_i * &vector).collect()
    }
}

impl Mul<&Vector> for SquareMatrix {
    type Output = Vector;
    fn mul(self, vector: &Vector) -> Self::Output {
        self.iter().map(|self_i| self_i * vector).collect()
    }
}

impl Add for SquareMatrix {
    type Output = Self;
    fn add(mut self, vector: Self) -> Self::Output {
        self += vector;
        self
    }
}

impl Add<&Self> for SquareMatrix {
    type Output = Self;
    fn add(mut self, vector: &Self) -> Self::Output {
        self += vector;
        self
    }
}

impl AddAssign for SquareMatrix {
    fn add_assign(&mut self, vector: Self) {
        self.iter_mut()
            .zip(vector.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry += tensor_rank_1);
    }
}

impl AddAssign<&Self> for SquareMatrix {
    fn add_assign(&mut self, vector: &Self) {
        self.iter_mut()
            .zip(vector.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry += tensor_rank_1);
    }
}

impl Mul for SquareMatrix {
    type Output = Self;
    fn mul(self, matrix: Self) -> Self::Output {
        let mut output = Self::zero(matrix.len());
        self.iter()
            .zip(output.iter_mut())
            .for_each(|(self_i, output_i)| {
                self_i
                    .iter()
                    .zip(matrix.iter())
                    .for_each(|(self_ij, matrix_j)| *output_i += matrix_j * self_ij)
            });
        output
    }
}

impl Sub for SquareMatrix {
    type Output = Self;
    fn sub(mut self, vector: Self) -> Self::Output {
        self -= vector;
        self
    }
}

impl Sub<&Self> for SquareMatrix {
    type Output = Self;
    fn sub(mut self, vector: &Self) -> Self::Output {
        self -= vector;
        self
    }
}

impl SubAssign for SquareMatrix {
    fn sub_assign(&mut self, vector: Self) {
        self.iter_mut()
            .zip(vector.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry -= tensor_rank_1);
    }
}

impl SubAssign<&Self> for SquareMatrix {
    fn sub_assign(&mut self, vector: &Self) {
        self.iter_mut()
            .zip(vector.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry -= tensor_rank_1);
    }
}
