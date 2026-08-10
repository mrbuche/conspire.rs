#[cfg(test)]
mod test;

mod ldl;
mod lu;

use crate::math::assert::FiniteDifference;

use crate::math::{
    Hessian, Rank2, Scalar, Tensor, TensorRank2Vec2D, TensorVec, Vector, write_tensor_rank_0,
};

use std::{
    fmt::{self, Display, Formatter},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    vec::IntoIter,
};

pub use ldl::LdlDecomposition;
pub use lu::LuDecomposition;

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
    pub fn zero(len: usize) -> Self {
        (0..len).map(|_| Vector::zero(len)).collect()
    }
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

impl<const D: usize, I, J> From<TensorRank2Vec2D<D, I, J>> for SquareMatrix {
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
    fn quadratic_form(&self, vector: &Vector) -> Scalar {
        self.iter()
            .zip(vector.iter())
            .map(|(self_i, vector_i)| vector_i * (self_i * vector))
            .sum()
    }
    fn retain_from(self, retained: &[bool]) -> SquareMatrix {
        self.into_iter()
            .zip(retained.iter())
            .filter(|(_, retained_i)| **retained_i)
            .map(|(row, _)| {
                row.into_iter()
                    .zip(retained.iter())
                    .filter(|(_, retained_j)| **retained_j)
                    .map(|(entry, _)| entry)
                    .collect()
            })
            .collect()
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
