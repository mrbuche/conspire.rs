#[cfg(test)]
mod test;

#[cfg(test)]
use crate::math::test::ErrorTensor;

use crate::{
    ABS_TOL,
    math::{
        Hessian, Rank2, Scalar, Tensor, TensorRank2Vec2D, TensorVec, Vector, write_tensor_rank_0,
    },
};
use std::{
    collections::VecDeque,
    fmt::{self, Display, Formatter},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    vec::IntoIter,
};

/// Rearrangement structure for a banded matrix.
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
        (0..num).for_each(|i| {
            (0..num)
                .filter(|&j| structure[i][j] && i != j)
                .for_each(|j| adj_list[i].push(j))
        });
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
            (i + 1..num)
                .filter(|&j| structure[mapping[i]][mapping[j]])
                .for_each(|j| bandwidth = bandwidth.max(j - i))
        });
        Self {
            bandwidth,
            mapping,
            inverse,
        }
    }
}

/// Possible errors for square matrices.
#[derive(Debug, PartialEq)]
pub enum SquareMatrixError {
    Singular,
}

/// A square matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct SquareMatrix(Vec<Vector>);

impl Default for SquareMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl SquareMatrix {
    // /// Solve a system of linear equations using the LDL decomposition.
    // pub fn solve_ldl(&mut self, b: &Vector) -> Result<Vector, SquareMatrixError> {
    //     let n = self.len();
    //     let mut p: Vec<usize> = (0..n).collect();
    //     let mut d: Vec<Scalar> = vec![0.0; n];
    //     let mut l: Vec<Vec<Scalar>> = vec![vec![0.0; n]; n];
    //     // for i in 0..n {
    //     //     for j in 0..n {
    //     //         assert!((self[i][j] - self[j][i]).abs() < ABS_TOL || (self[i][j] / self[j][i] - 1.0).abs() < ABS_TOL)
    //     //     }
    //     // }
    //     for i in 0..n {
    //         let mut max_row = i;
    //         let mut max_val = self[max_row][i].abs();
    //         for k in i + 1..n {
    //             if self[k][i].abs() > max_val {
    //                 max_row = k;
    //                 max_val = self[max_row][i].abs();
    //             }
    //         }
    //         if max_row != i {
    //             self.0.swap(i, max_row);
    //             p.swap(i, max_row);
    //         }
    //         let mut sum = 0.0;
    //         for k in 0..i {
    //             sum += l[i][k] * d[k] * l[i][k];
    //         }
    //         let pivot = self[i][i] - sum;
    //         if pivot.abs() < ABS_TOL {
    //             return Err(SquareMatrixError::Singular);
    //         }
    //         d[i] = pivot;
    //         l[i][i] = 1.0;
    //         for j in i + 1..n {
    //             sum = 0.0;
    //             for k in 0..i {
    //                 sum += l[j][k] * d[k] * l[i][k];
    //             }
    //             l[j][i] = (self[j][i] - sum) / d[i];
    //         }
    //     }
    //     let mut y = Vector::zero(n);
    //     for i in 0..n {
    //         y[i] = b[p[i]];
    //         for j in 0..i {
    //             y[i] -= l[i][j] * y[j];
    //         }
    //     }
    //     let mut x = Vector::zero(n);
    //     for i in 0..n {
    //         x[i] = y[i] / d[i];
    //     }
    //     for i in (0..n).rev() {
    //         for j in i + 1..n {
    //             x[i] -= l[j][i] * x[j];
    //         }
    //     }
    //     // Ok(x)
    //     let mut xs = Vector::zero(n);
    //     for i in 0..n {
    //         xs[p[i]] = x[i]
    //     }
    //     Ok(xs)
    //     // let mut p_reverse = vec![0; n];
    //     // for (i, &pi) in p.iter().enumerate() {
    //     //     p_reverse[pi] = i;
    //     // }
    //     // let mut xs = Vector::zero(n);
    //     // for i in 0..n {
    //     //     // xs[i] = x[p_reverse[i]]
    //     //     xs[p_reverse[i]] = x[i]
    //     // }
    //     // Ok(xs)
    // }
    /// Solve a system of linear equations using the LU decomposition.
    pub fn solve_lu(&self, b: &Vector) -> Result<Vector, SquareMatrixError> {
        let n = self.len();
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
                return Err(SquareMatrixError::Singular);
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
        let mut x: Vector = p.into_iter().map(|p_i| b[p_i]).collect();
        forward_substitution(&mut x, &lu);
        backward_substitution(&mut x, &lu);
        Ok(x)
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
        let mut max_val;
        let mut pivot;
        let mut rearr: Self = (0..n)
            .map(|i| (0..n).map(|j| self[banded.old(i)][banded.old(j)]).collect())
            .collect();
        for i in 0..n {
            end = n.min(i + 1 + bandwidth);
            pivot = rearr[i][i];
            if pivot.abs() < ABS_TOL {
                max_row = i;
                max_val = rearr[max_row][i].abs();
                for k in i + 1..end {
                    if rearr[k][i].abs() > max_val {
                        max_row = k;
                        max_val = rearr[max_row][i].abs();
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
        let mut x: Vector = p.into_iter().map(|p_i| b[banded.old(p_i)]).collect();
        forward_substitution(&mut x, &rearr);
        backward_substitution(&mut x, &rearr);
        Ok((0..n).map(|i| x[banded.map(i)]).collect())
    }
    pub fn zero(len: usize) -> Self {
        (0..len).map(|_| Vector::zero(len)).collect()
    }
}

fn forward_substitution(x: &mut Vector, a: &SquareMatrix) {
    a.iter().enumerate().for_each(|(i, a_i)| {
        x[i] -= a_i
            .iter()
            .take(i)
            .zip(x.iter().take(i))
            .map(|(a_ij, x_j)| a_ij * x_j)
            .sum::<Scalar>()
    })
}

fn backward_substitution(x: &mut Vector, a: &SquareMatrix) {
    a.0.iter().enumerate().rev().for_each(|(i, a_i)| {
        x[i] -= a_i
            .iter()
            .skip(i + 1)
            .zip(x.iter().skip(i + 1))
            .map(|(a_ij, x_j)| a_ij * x_j)
            .sum::<Scalar>();
        x[i] /= a_i[i];
    })
}

#[cfg(test)]
impl ErrorTensor for SquareMatrix {
    fn error_fd(&self, comparator: &Self, epsilon: &Scalar) -> Option<(bool, usize)> {
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

impl<const N: usize> From<[[Scalar; N]; N]> for SquareMatrix {
    fn from(array: [[Scalar; N]; N]) -> Self {
        array.into_iter().map(Vector::from).collect()
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
}

impl Rank2 for SquareMatrix {
    type Transpose = Self;
    fn deviatoric(&self) -> Self {
        let len = self.len();
        let scale = -self.trace() / len as Scalar;
        (0..len)
            .map(|i| {
                (0..len)
                    .map(|j| ((i == j) as u8) as Scalar * scale)
                    .collect()
            })
            .collect::<Self>()
            + self
    }
    fn deviatoric_and_trace(&self) -> (Self, Scalar) {
        let len = self.len();
        let trace = self.trace();
        let scale = -trace / len as Scalar;
        (
            (0..len)
                .map(|i| {
                    (0..len)
                        .map(|j| ((i == j) as u8) as Scalar * scale)
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
        self.iter().enumerate().all(|(i, self_i)| {
            self_i
                .iter()
                .enumerate()
                .all(|(j, self_ij)| self_ij == &((i == j) as u8 as Scalar))
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
    fn squared_trace(&self) -> Scalar {
        self.iter()
            .enumerate()
            .map(|(i, self_i)| {
                self_i
                    .iter()
                    .zip(self.iter())
                    .map(|(self_ij, self_j)| self_ij * self_j[i])
                    .sum::<Scalar>()
            })
            .sum()
    }
    fn trace(&self) -> Scalar {
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
    fn len(&self) -> usize {
        self.0.len()
    }
    fn size(&self) -> usize {
        unimplemented!("Do not like that inner Vecs could be different sizes")
    }
}

impl IntoIterator for SquareMatrix {
    type Item = Vector;
    type IntoIter = IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl TensorVec for SquareMatrix {
    type Item = Vector;
    fn append(&mut self, other: &mut Self) {
        self.0.append(&mut other.0)
    }
    fn capacity(&self) -> usize {
        self.0.capacity()
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn new() -> Self {
        Self(Vec::new())
    }
    fn push(&mut self, item: Self::Item) {
        self.0.push(item)
    }
    fn remove(&mut self, index: usize) -> Self::Item {
        self.0.remove(index)
    }
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&Self::Item) -> bool,
    {
        self.0.retain(f)
    }
    fn swap_remove(&mut self, index: usize) -> Self::Item {
        self.0.swap_remove(index)
    }
}

impl Sum for SquareMatrix {
    fn sum<Ii>(iter: Ii) -> Self
    where
        Ii: Iterator<Item = Self>,
    {
        iter.reduce(|mut acc, item| {
            acc += item;
            acc
        })
        .unwrap_or_else(Self::default)
    }
}

impl Div<Scalar> for SquareMatrix {
    type Output = Self;
    fn div(mut self, tensor_rank_0: Scalar) -> Self::Output {
        self /= &tensor_rank_0;
        self
    }
}

impl Div<&Scalar> for SquareMatrix {
    type Output = Self;
    fn div(mut self, tensor_rank_0: &Scalar) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl DivAssign<Scalar> for SquareMatrix {
    fn div_assign(&mut self, tensor_rank_0: Scalar) {
        self.iter_mut().for_each(|entry| *entry /= &tensor_rank_0);
    }
}

impl DivAssign<&Scalar> for SquareMatrix {
    fn div_assign(&mut self, tensor_rank_0: &Scalar) {
        self.iter_mut().for_each(|entry| *entry /= tensor_rank_0);
    }
}

impl Mul<Scalar> for SquareMatrix {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: Scalar) -> Self::Output {
        self *= &tensor_rank_0;
        self
    }
}

impl Mul<&Scalar> for SquareMatrix {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &Scalar) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl Mul<&Scalar> for &SquareMatrix {
    type Output = SquareMatrix;
    fn mul(self, tensor_rank_0: &Scalar) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_0).collect()
    }
}

impl MulAssign<Scalar> for SquareMatrix {
    fn mul_assign(&mut self, tensor_rank_0: Scalar) {
        self.iter_mut().for_each(|entry| *entry *= &tensor_rank_0);
    }
}

impl MulAssign<&Scalar> for SquareMatrix {
    fn mul_assign(&mut self, tensor_rank_0: &Scalar) {
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
    fn sub(mut self, square_matrix: Self) -> Self::Output {
        self -= square_matrix;
        self
    }
}

impl Sub<&Self> for SquareMatrix {
    type Output = Self;
    fn sub(mut self, square_matrix: &Self) -> Self::Output {
        self -= square_matrix;
        self
    }
}

impl Sub for &SquareMatrix {
    type Output = SquareMatrix;
    fn sub(self, square_matrix: Self) -> Self::Output {
        square_matrix
            .iter()
            .zip(self.iter())
            .map(|(square_matrix_i, self_i)| self_i - square_matrix_i)
            .collect()
    }
}

impl SubAssign for SquareMatrix {
    fn sub_assign(&mut self, square_matrix: Self) {
        self.iter_mut()
            .zip(square_matrix.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry -= tensor_rank_1);
    }
}

impl SubAssign<&Self> for SquareMatrix {
    fn sub_assign(&mut self, square_matrix: &Self) {
        self.iter_mut()
            .zip(square_matrix.iter())
            .for_each(|(self_entry, tensor_rank_1)| *self_entry -= tensor_rank_1);
    }
}
