#[cfg(test)]
mod test;

use crate::math::assert::FiniteDifference;

use crate::{
    ABS_TOL, REL_TOL,
    math::{
        Hessian, Rank2, Scalar, Tensor, TensorRank2Vec2D, TensorVec, Vector, simd,
        write_tensor_rank_0,
    },
};
use std::{
    fmt::{self, Display, Formatter},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    vec::IntoIter,
};

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
    /// Factorize the matrix using the LU decomposition.
    pub fn factorize_lu(&self) -> Result<LuDecomposition, SquareMatrixError> {
        let mut decomposition = LuDecomposition {
            lu: self.clone(),
            permutation: (0..self.len()).collect(),
        };
        decomposition.factorize()?;
        Ok(decomposition)
    }
    /// Factorize the matrix into an existing LU decomposition of the same size.
    pub fn factorize_lu_into(
        &self,
        decomposition: &mut LuDecomposition,
    ) -> Result<(), SquareMatrixError> {
        let LuDecomposition { lu, permutation } = decomposition;
        lu.iter_mut().zip(self.iter()).for_each(|(lu_i, self_i)| {
            lu_i.iter_mut()
                .zip(self_i.iter())
                .for_each(|(a, b)| *a = *b)
        });
        permutation
            .iter_mut()
            .enumerate()
            .for_each(|(i, permutation_i)| *permutation_i = i);
        decomposition.factorize()
    }
    /// Solve a system of linear equations using the LU decomposition.
    pub fn solve_lu(&self, b: &Vector) -> Result<Vector, SquareMatrixError> {
        Ok(self.factorize_lu()?.solve(b))
    }
    pub fn zero(len: usize) -> Self {
        (0..len).map(|_| Vector::zero(len)).collect()
    }
}

/// The LU decomposition of a square matrix.
pub struct LuDecomposition {
    lu: SquareMatrix,
    permutation: Vec<usize>,
}

impl LuDecomposition {
    fn factorize(&mut self) -> Result<(), SquareMatrixError> {
        let Self { lu, permutation } = self;
        let n = lu.len();
        //
        // A pivot is singular only once it is negligible both absolutely and
        // against the largest pivot so far, so that scaling the matrix down
        // cannot alone condemn it, nor a wide but honest spread of pivots.
        //
        let mut largest = 0.0;
        for i in 0..n {
            let mut max_row = i;
            let mut max_val = lu[i][i].abs();
            for k in i + 1..n {
                let candidate = lu[k][i].abs();
                if candidate > max_val {
                    max_row = k;
                    max_val = candidate;
                }
            }
            if max_row != i {
                lu.0.swap(i, max_row);
                permutation.swap(i, max_row);
            }
            largest = max_val.max(largest);
            if max_val < ABS_TOL && max_val <= REL_TOL * largest {
                return Err(SquareMatrixError::Singular);
            }
            let pivot = lu[i][i];
            let (front, back) = lu.0.split_at_mut(i + 1);
            let column = &front[i].as_slice()[i + 1..];
            let mut count = 0;
            for row in 0..back.len() {
                let factor = back[row][i];
                if factor != 0.0 {
                    back[row][i] = factor / pivot;
                    if row != count {
                        back.swap(row, count);
                        permutation.swap(i + 1 + row, i + 1 + count)
                    }
                    count += 1
                }
            }
            if column.len() < 4 {
                back[..count].iter_mut().for_each(|row| {
                    let factor = row[i];
                    row.as_mut_slice()[i + 1..]
                        .iter_mut()
                        .zip(column.iter())
                        .for_each(|(row_k, column_k)| *row_k -= factor * column_k)
                })
            } else {
                back[..count].chunks_mut(4).for_each(|chunk| {
                    if let [a, b, c, d] = chunk {
                        let u = [-a[i], -b[i], -c[i], -d[i]];
                        simd::rank_one_quad(
                            &mut a.as_mut_slice()[i + 1..],
                            &mut b.as_mut_slice()[i + 1..],
                            &mut c.as_mut_slice()[i + 1..],
                            &mut d.as_mut_slice()[i + 1..],
                            column,
                            u,
                        )
                    } else {
                        chunk.iter_mut().for_each(|row| {
                            let factor = row[i];
                            simd::axpy(&mut row.as_mut_slice()[i + 1..], column, factor)
                        })
                    }
                })
            }
        }
        Ok(())
    }
    /// An unfactorized decomposition sized to hold that of a matrix of the given length.
    pub fn zero(len: usize) -> Self {
        Self {
            lu: SquareMatrix::zero(len),
            permutation: (0..len).collect(),
        }
    }
    /// Solve a system of linear equations for another right-hand side.
    pub fn solve(&self, b: &Vector) -> Vector {
        let mut x = Vector::zero(self.permutation.len());
        self.solve_into(b, &mut x);
        x
    }
    /// Solve a system of linear equations into an existing vector.
    pub fn solve_into(&self, b: &Vector, x: &mut Vector) {
        self.permutation
            .iter()
            .zip(x.iter_mut())
            .for_each(|(&p_i, x_i)| *x_i = b[p_i]);
        forward_substitution(x, &self.lu);
        backward_substitution(x, &self.lu)
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

impl FiniteDifference for SquareMatrix {
    fn error_fd(&self, comparator: &Self, epsilon: Scalar) -> Option<(bool, usize)> {
        let error_count = self
            .iter()
            .zip(comparator.iter())
            .map(|(self_i, comparator_i)| {
                self_i
                    .iter()
                    .zip(comparator_i.iter())
                    .filter(|&(&self_ij, &comparator_ij)| {
                        (self_ij / comparator_ij - 1.0).abs() >= epsilon
                            && (self_ij.abs() >= epsilon || comparator_ij.abs() >= epsilon)
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

impl From<SquareMatrix> for Vec<Vec<Scalar>> {
    fn from(square_matrix: SquareMatrix) -> Self {
        square_matrix
            .into_iter()
            .map(|vector| vector.into())
            .collect()
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
    fn entry(&self, row: usize, column: usize) -> Scalar {
        self[row][column]
    }
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
    fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional)
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
    fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
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
