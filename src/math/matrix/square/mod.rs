#[cfg(test)]
mod test;

#[cfg(test)]
use crate::math::test::ErrorTensor;

use crate::{
    ABS_TOL,
    math::{
        Hessian, Matrix, Rank2, Tensor, TensorRank0, TensorRank2Vec2D, TensorVec, Vector,
        tensor::TensorError, write_tensor_rank_0,
    },
};
use std::{
    collections::VecDeque,
    fmt::{self, Display, Formatter},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

/// ???
pub struct Banded {
    bandwidth: usize,
    inverse: Vec<usize>,
    mapping: Vec<usize>,
}

impl Banded {
    pub fn map(&self, old: usize) -> usize {
        self.inverse[old]
    }
    pub fn old(&self, new: usize) -> usize {
        self.mapping[new]
    }
    pub fn width(&self) -> usize {
        self.bandwidth
    }
}

impl From<Vec<Vec<bool>>> for Banded {
    fn from(structure: Vec<Vec<bool>>) -> Self {
        let num = structure.len();
        structure.iter().enumerate().for_each(|(i, row_i)| {
            assert_eq!(row_i.len(), num);
            row_i
                .iter()
                .zip(structure.iter())
                .for_each(|(entry_ij, row_j)| assert_eq!(&row_j[i], entry_ij))
        });
        let mut adj_list = vec![Vec::new(); num];
        for i in 0..num {
            for j in 0..num {
                if structure[i][j] && i != j {
                    adj_list[i].push(j);
                }
            }
        }
        let start_vertex = (0..num)
            .min_by_key(|&i| adj_list[i].len())
            .expect("Matrix must have at least one entry.");
        let mut visited = vec![false; num];
        let mut mapping = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start_vertex);
        visited[start_vertex] = true;
        while let Some(vertex) = queue.pop_front() {
            mapping.push(vertex);
            let mut neighbors: Vec<usize> = adj_list[vertex]
                .iter()
                .filter(|&&neighbor| !visited[neighbor])
                .copied()
                .collect();
            neighbors.sort_by_key(|&neighbor| adj_list[neighbor].len());
            for neighbor in neighbors {
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
        mapping.reverse();
        let mut inverse = vec![0; num];
        mapping
            .iter()
            .enumerate()
            .for_each(|(new, &old)| inverse[old] = new);
        let mut bandwidth = 0;
        (0..num).for_each(|i| {
            (i + 1..num).for_each(|j| {
                if structure[mapping[i]][mapping[j]] {
                    bandwidth = bandwidth.max(j - i)
                }
            })
        });
        Self {
            bandwidth,
            mapping,
            inverse,
        }
    }
}

/// Possible errors for square matrices.
#[derive(PartialEq)]
pub enum SquareMatrixError {
    Singular,
}

/// A square matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct SquareMatrix(Vec<Vector>);

impl SquareMatrix {
    /// Solve a system of linear equations using the LU decomposition.
    pub fn solve_lu(&mut self, b: &Vector) -> Result<Vector, SquareMatrixError> {
        let n = self.len();
        let mut p: Vec<usize> = (0..n).collect();
        let mut factor;
        let mut max_row;
        let mut pivot;
        let mut track;
        for i in 0..n {
            max_row = i;
            track = self[max_row][i].abs();
            (i + 1..n).for_each(|k| {
                if self[k][i].abs() > track {
                    max_row = k;
                    track = self[max_row][i].abs();
                }
            });
            if max_row != i {
                self.0.swap(i, max_row);
                p.swap(i, max_row);
            }
            pivot = self[i][i];
            if pivot.abs() < ABS_TOL {
                return Err(SquareMatrixError::Singular);
            }
            for j in i + 1..n {
                self[j][i] /= pivot;
                factor = self[j][i];
                for k in i + 1..n {
                    self[j][k] -= factor * self[i][k];
                }
            }
        }
        let mut xy: Vector = p.into_iter().map(|p_i| b[p_i]).collect();
        (0..n).for_each(|i| {
            xy[i] -= self[i]
                .iter()
                .take(i)
                .zip(xy.iter())
                .map(|(lu_ij, y_j)| lu_ij * y_j)
                .sum::<TensorRank0>()
        });
        (0..n).rev().for_each(|i| {
            xy[i] -= self[i]
                .iter()
                .skip(i + 1)
                .zip(xy.iter().skip(i + 1))
                .map(|(lu_ij, x_j)| lu_ij * x_j)
                .sum::<TensorRank0>();
            xy[i] /= self[i][i];
        });
        Ok(xy)
    }
    /// Solve a system of linear equations rearranged in a banded structure using the LU decomposition.
    pub fn solve_lu_banded(
        &self,
        b: &Vector,
        banded: &Banded,
    ) -> Result<Vector, SquareMatrixError> {
        let bandwidth = banded.width();
        let mut bandwidth_updated;
        let n = self.len();
        let mut p: Vec<usize> = (0..n).collect();
        let mut end;
        let mut factor;
        let mut max_row;
        let mut pivot;
        let mut track;
        let mut rearr: Self = (0..n)
            .map(|i| (0..n).map(|j| self[banded.old(i)][banded.old(j)]).collect())
            .collect();
        for i in 0..n {
            end = n.min(i + 1 + bandwidth);
            pivot = rearr[i][i];
            if pivot.abs() < ABS_TOL {
                max_row = i;
                track = rearr[max_row][i].abs();
                for k in i + 1..end {
                    if rearr[k][i].abs() > track {
                        max_row = k;
                        track = rearr[max_row][i].abs();
                    }
                }
                if max_row != i {
                    rearr.0.swap(i, max_row);
                    p.swap(i, max_row);
                    pivot = rearr[i][i];
                    if pivot.abs() < ABS_TOL {
                        return Err(SquareMatrixError::Singular);
                    }
                }
            }
            bandwidth_updated = bandwidth;
            (i + 1 + bandwidth..n).for_each(|j| {
                if rearr[i][j] != 0.0 {
                    // if j - i > bandwidth_updated {
                    //     bandwidth_updated = j - 1
                    // }
                    bandwidth_updated = bandwidth_updated.max(j - i)
                }
            });
            end = n.min(i + 1 + bandwidth_updated);
            for j in i + 1..end {
                if rearr[j][i] != 0.0 {
                    rearr[j][i] /= pivot;
                    factor = rearr[j][i];
                    for k in i + 1..end {
                        rearr[j][k] -= factor * rearr[i][k];
                    }
                }
            }
        }
        let mut xy: Vector = p.into_iter().map(|p_i| b[banded.old(p_i)]).collect();
        (0..n).for_each(|i| {
            xy[i] -= rearr[i]
                .iter()
                .take(i)
                .zip(xy.iter())
                .map(|(lu_ij, y_j)| lu_ij * y_j)
                .sum::<TensorRank0>()
        });
        (0..n).rev().for_each(|i| {
            xy[i] -= rearr[i]
                .iter()
                .skip(i + 1)
                .zip(xy.iter().skip(i + 1))
                .map(|(lu_ij, x_j)| lu_ij * x_j)
                .sum::<TensorRank0>();
            xy[i] /= rearr[i][i];
        });
        Ok((0..n).map(|i| xy[banded.map(i)]).collect())
    }
    /// Verifies a minimum using the null space of the constraint matrix.
    pub fn verify(self, null_space: Matrix) -> bool {
        // Supposed to use the Hessian but should be OK since Z will hit None.
        let mut result = Self::zero(null_space.width());
        null_space
            .transpose()
            .iter()
            .zip(result.iter_mut())
            .for_each(|(zt_i, res_i)| {
                zt_i.iter().zip(self.iter()).for_each(|(zt_ij, self_j)| {
                    null_space
                        .iter()
                        .zip(self_j.iter())
                        .for_each(|(z_k, self_jk)| {
                            z_k.iter()
                                .zip(res_i.iter_mut())
                                .for_each(|(z_kl, res_il)| *res_il += zt_ij * self_jk * z_kl)
                        })
                })
            });
        result.is_positive_definite()
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

impl Display for SquareMatrix {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
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
        let mut square_matrix = Self::zero(tensor_rank_2_vec_2d.len() * D);
        tensor_rank_2_vec_2d
            .iter()
            .enumerate()
            .for_each(|(a, entry_a)| {
                entry_a.iter().enumerate().for_each(|(b, entry_ab)| {
                    entry_ab.iter().enumerate().for_each(|(i, entry_ab_i)| {
                        entry_ab_i.iter().enumerate().for_each(|(j, entry_ab_ij)| {
                            square_matrix[D * a + i][D * b + j] = *entry_ab_ij
                        })
                    })
                })
            });
        square_matrix
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
