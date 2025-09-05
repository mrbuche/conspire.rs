#[cfg(test)]
mod test;

use crate::{ABS_TOL, math::{Banded, Scalar, TensorVec, Vector, write_tensor_rank_0}};
use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
    ops::{Index, Mul},
};
use super::SquareMatrixError;

/// A sparse matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseSquareMatrix {
    len: usize,
    values: Vec<Scalar>,
    col_indices: Vec<usize>,
    row_pointers: Vec<usize>,
}

impl SparseSquareMatrix {
    /// Adds a value to the sparse matrix at the specified row and column.
    pub fn add_value(&mut self, row: usize, col: usize, val: Scalar) {
        self.values.push(val);
        self.col_indices.push(col);
        (row + 1..=self.len).for_each(|i| self.row_pointers[i] += 1)
    }
    /// Creates a new sparse matrix with the given dimensions.
    pub fn new(len: usize) -> Self {
        Self {
            len,
            values: Vec::new(),
            col_indices: Vec::new(),
            row_pointers: vec![0; len + 1],
        }
    }
    // /// Solve a system of linear equations using the LU decomposition.
    // pub fn solve_lu(&self, b: &Vector) -> Vector {
    // // pub fn solve_lu(&self, b: &Vector) -> Result<Vector, SquareMatrixError> {
    //     assert!(self.len > 0, "Matrix must be non-empty for LU decomposition");
    //     assert!(b.len() == self.len, "Vector b must match matrix dimensions");

    //     // Permutation vector for row swaps.
    //     let mut p: Vec<usize> = (0..self.len).collect();

    //     // Temporary storage for L and U matrices in CSR format.
    //     let mut l_values = Vec::new();
    //     let mut l_col_indices = Vec::new();
    //     let mut l_row_pointers = vec![0; self.len + 1];

    //     let mut u_values = Vec::new();
    //     let mut u_col_indices = Vec::new();
    //     let mut u_row_pointers = vec![0; self.len + 1];

    //     // Perform LU decomposition.
    //     for row in 0..self.len {// Partial pivoting: Find the row with the largest absolute value in column `row`.
    //         let mut max_row = row;
    //         let mut max_val = Scalar::default();

    //         // for r in row..self.len {
    //         //     let start = self.row_pointers[p[r]];
    //         //     let end = self.row_pointers[p[r] + 1];
    //         //     for i in start..end {
    //         //         if self.col_indices[i] == row {
    //         //             let value = self.values[i];
    //         //             if value.abs() > max_val {
    //         //                 max_row = r;
    //         //                 max_val = value.abs();
    //         //             }
    //         //         }
    //         //     }
    //         // }
    //         for i in k..self.len {
    //             let row_idx = p[i];
    //             if let Some(&value) = working_matrix[row_idx].get(&k) {
    //                 if value.abs() > max_val {
    //                     max_val = value.abs();
    //                     pivot_row = i;
    //                 }
    //             }
    //         }

    //         // Swap rows in the permutation vector.
    //         if max_row != row {
    //             p.swap(row, max_row);
    //         }

    //         // Check for singularity.
    //         if max_val < ABS_TOL {
    //             panic!("Matrix is singular or poorly conditioned");
    //         }

    //         let mut u_row_map: HashMap<usize, Scalar> = HashMap::new();
    //         let mut l_row_map: HashMap<usize, Scalar> = HashMap::new();

    //         let start = self.row_pointers[p[row]];
    //         let end = self.row_pointers[p[row] + 1];

    //         // Populate U row map with the current row of A.
    //         for i in start..end {
    //             let col = self.col_indices[i];
    //             let value = self.values[i];
    //             u_row_map.insert(col, value);
    //         }

    //         // Eliminate elements below the diagonal.
    //         for j in 0..row {
    //             let l_start = l_row_pointers[j];
    //             let l_end = l_row_pointers[j + 1];

    //             let mut l_factor = Scalar::default();
    //             for k in l_start..l_end {
    //                 if l_col_indices[k] == row {
    //                     l_factor = l_values[k];
    //                     break;
    //                 }
    //             }

    //             // if l_factor != Scalar::default() {
    //             //     for (&col, &value) in u_row_map.iter() {
    //             //         if col >= row {
    //             //             u_row_map.entry(col).and_modify(|v| *v -= l_factor * value);
    //             //         }
    //             //     }
    //             // }
    //             if l_factor != Scalar::default() {
    //                 // Collect the keys and values to modify into a temporary vector.
    //                 let updates: Vec<(usize, Scalar)> = u_row_map
    //                     .iter()
    //                     .filter(|(col, _)| col >= &&row)
    //                     .map(|(&col, &value)| (col, value))
    //                     .collect();

    //                 // Perform the updates on u_row_map.
    //                 for (col, value) in updates {
    //                     u_row_map.entry(col).and_modify(|v| *v -= l_factor * value);
    //                 }
    //             }
    //         }

    //         // Extract U row and update L.
    //         for (&col, &value) in u_row_map.iter() {
    //             if col == row {
    //                 // Diagonal element goes to U.
    //                 u_values.push(value);
    //                 u_col_indices.push(col);
    //             } else if col > row {
    //                 // Upper triangular element goes to U.
    //                 u_values.push(value);
    //                 u_col_indices.push(col);
    //             } else {
    //                 // Lower triangular element goes to L.
    //                 l_row_map.insert(col, value / u_row_map[&row]);
    //             }
    //         }

    //         for (&col, &value) in l_row_map.iter() {
    //             l_values.push(value);
    //             l_col_indices.push(col);
    //         }

    //         l_row_pointers[row + 1] = l_values.len();
    //         u_row_pointers[row + 1] = u_values.len();
    //     }

    //     // Construct L and U matrices in CSR format.
    //     let l = SparseSquareMatrix {
    //         len: self.len,
    //         values: l_values,
    //         col_indices: l_col_indices,
    //         row_pointers: l_row_pointers,
    //     };

    //     let u = SparseSquareMatrix {
    //         len: self.len,
    //         values: u_values,
    //         col_indices: u_col_indices,
    //         row_pointers: u_row_pointers,
    //     };

    //     println!("{}", l);
    //     println!("{}", u);

    //     // Apply the permutation vector to b.
    //     let mut permuted_b = Vector::zero(b.len());
    //     for i in 0..self.len {
    //         permuted_b[i] = b[p[i]];
    //     }

    //     // Step 1: Forward substitution to solve L * y = b.
    //     let mut y = Vector::zero(permuted_b.len());
    //     for row in 0..self.len {
    //         let start = l.row_pointers[row];
    //         let end = l.row_pointers[row + 1];

    //         let mut sum = Scalar::default();
    //         for i in start..end {
    //             let col = l.col_indices[i];
    //             if col < row {
    //                 sum += l.values[i] * y[col];
    //             }
    //         }

    //         y[row] = (permuted_b[row] - sum) / l[[row, row]];
    //     }

    //     // Step 2: Backward substitution to solve U * x = y.
    //     let mut x = Vector::zero(y.len());
    //     for row in (0..self.len).rev() {
    //         let start = u.row_pointers[row];
    //         let end = u.row_pointers[row + 1];

    //         let mut sum = Scalar::default();
    //         for i in start..end {
    //             let col = u.col_indices[i];
    //             if col > row {
    //                 sum += u.values[i] * x[col];
    //             }
    //         }

    //         x[row] = (y[row] - sum) / u[[row, row]];
    //     }

    //     x
    // }
    /// Solve a system of linear equations using sparse banded LU decomposition with partial pivoting.
    pub fn solve_lu_banded(
        &self,
        b: &Vector,
        banded: &Banded,
    ) -> Result<Vector, SquareMatrixError> {
        let bandwidth = banded.width();
        let mut bandwidth_updated = bandwidth;
        let n = self.len;
        let mut p: Vec<usize> = (0..n).collect();
        
        // Create sparse working matrix using HashMap for easy manipulation
        // We'll store the rearranged matrix in sparse format
        let mut working_matrix = vec![HashMap::<usize, Scalar>::new(); n];
        
        // Initialize working matrix with rearranged data (banded ordering)
        for i in 0..n {
            let old_i = banded.old(i);
            let start = self.row_pointers[old_i];
            let end = self.row_pointers[old_i + 1];
            
            for idx in start..end {
                let old_j = self.col_indices[idx];
                // Find the new column index after banding rearrangement
                // if let Some(new_j) = banded.try_map(old_j) {
                //     working_matrix[i].insert(new_j, self.values[idx]);
                // }
                let new_j = banded.map(old_j);
                working_matrix[i].insert(new_j, self.values[idx]);
            }
        }
        
        // LU decomposition with partial pivoting for banded sparse matrix
        for i in 0..n {
            let end = n.min(i + 1 + bandwidth_updated);
            
            // Get pivot element
            let mut pivot = working_matrix[i].get(&i).copied().unwrap_or(Scalar::default());
            
            // Partial pivoting if needed
            if pivot.abs() < ABS_TOL {
                let mut max_row = i;
                let mut max_val = pivot.abs();
                
                for k in (i + 1)..end {
                    let val = working_matrix[k].get(&i).copied().unwrap_or(Scalar::default());
                    if val.abs() > max_val {
                        max_row = k;
                        max_val = val.abs();
                    }
                }
                
                if max_row != i {
                    // Swap rows in working matrix
                    working_matrix.swap(i, max_row);
                    p.swap(i, max_row);
                    pivot = working_matrix[i].get(&i).copied().unwrap_or(Scalar::default());
                    
                    if pivot.abs() < ABS_TOL {
                        return Err(SquareMatrixError::Singular);
                    }
                }
            }
            
            // Update bandwidth if matrix has elements beyond current bandwidth
            let row_data = &working_matrix[i];
            for &col in row_data.keys() {
                if col > i {
                    bandwidth_updated = bandwidth_updated.max(col - i);
                }
            }
            let end = n.min(i + 1 + bandwidth_updated);
            
            // Gaussian elimination
            for j in (i + 1)..end {
                let factor_opt = working_matrix[j].get(&i).copied();
                
                if let Some(a_ji) = factor_opt {
                    if a_ji.abs() > ABS_TOL {
                        let factor = a_ji / pivot;
                        
                        // Store factor in L (we'll overwrite the lower triangular part)
                        working_matrix[j].insert(i, factor);
                        
                        // Collect all elements in row i that we need to use for elimination
                        let row_i_elements: Vec<(usize, Scalar)> = working_matrix[i]
                            .iter()
                            .filter(|&(col, _)| col > &i && col < &end)
                            .map(|(&col, &val)| (col, val))
                            .collect();
                        
                        // Update row j: A[j,k] -= factor * A[i,k] for k > i
                        for (col, a_ik) in row_i_elements {
                            let old_val = working_matrix[j].get(&col).copied().unwrap_or(Scalar::default());
                            let new_val = old_val - factor * a_ik;
                            
                            if new_val.abs() < ABS_TOL {
                                working_matrix[j].remove(&col);
                            } else {
                                working_matrix[j].insert(col, new_val);
                            }
                        }
                    }
                }
            }
        }
        
        // Apply permutation to b and convert to banded ordering
        let mut x: Vector = p.into_iter()
            .map(|p_i| b[banded.old(p_i)])
            .collect();
        
        // Forward substitution: L * y = x (in-place)
        forward_substitution_sparse(&mut x, &working_matrix);
        
        // Backward substitution: U * x = y (in-place) 
        backward_substitution_sparse(&mut x, &working_matrix);
        
        // Convert back from banded ordering to original ordering
        Ok((0..n).map(|i| x[banded.map(i)]).collect())
    }
    /// Solve a system of linear equations using sparse banded LU decomposition with partial pivoting.
    pub fn solve_lu_banded_new(
        &self,
        b: &Vector,
        banded: &Banded,
    ) -> Result<Vector, SquareMatrixError> {
        let bandwidth = banded.width();
        let n = self.len;
        let mut p: Vec<usize> = (0..n).collect();
        
        // For very sparse matrices, consider using HashMap for temporary storage
        let mut sparse_lu: HashMap<(usize, usize), Scalar> = HashMap::new();
        
        // Populate with non-zero entries from rearranged matrix
        for i in 0..n {
            for j in 0..n {
                let val = self[[banded.old(i), banded.old(j)]];
                if val.abs() > ABS_TOL * 0.1 {
                    sparse_lu.insert((i, j), val);
                }
            }
        }
        
        // LU decomposition (similar to before but using HashMap)
        for i in 0..n {
            let end = (i + 1 + bandwidth).min(n);
            let pivot = *sparse_lu.get(&(i, i)).unwrap_or(&0.0);
            
            if pivot.abs() < ABS_TOL {
                // Find pivot (similar logic as before)
                let mut max_row = i;
                let mut max_val = pivot.abs();
                
                for k in (i + 1)..end {
                    let val = sparse_lu.get(&(k, i)).unwrap_or(&0.0).abs();
                    if val > max_val {
                        max_val = val;
                        max_row = k;
                    }
                }
                
                if max_row != i {
                    // Swap rows in HashMap
                    for j in 0..n {
                        let val_i = sparse_lu.remove(&(i, j)).unwrap_or(0.0);
                        let val_max = sparse_lu.remove(&(max_row, j)).unwrap_or(0.0);
                        if val_i.abs() > ABS_TOL { sparse_lu.insert((max_row, j), val_i); }
                        if val_max.abs() > ABS_TOL { sparse_lu.insert((i, j), val_max); }
                    }
                    p.swap(i, max_row);
                }
            }
            
            let pivot_val = *sparse_lu.get(&(i, i)).unwrap_or(&0.0);
            if pivot_val.abs() < ABS_TOL {
                return Err(SquareMatrixError::Singular);
            }
            
            // Elimination step
            for j in (i + 1)..end {
                let elem_ji = *sparse_lu.get(&(j, i)).unwrap_or(&0.0);
                if elem_ji.abs() > ABS_TOL {
                    let factor = elem_ji / pivot_val;
                    sparse_lu.insert((j, i), factor);
                    
                    for k in (i + 1)..(i + 1 + bandwidth).min(n) {
                        let elem_ik = *sparse_lu.get(&(i, k)).unwrap_or(&0.0);
                        if elem_ik.abs() > ABS_TOL {
                            let elem_jk = *sparse_lu.get(&(j, k)).unwrap_or(&0.0);
                            let new_val = elem_jk - factor * elem_ik;
                            if new_val.abs() > ABS_TOL {
                                sparse_lu.insert((j, k), new_val);
                            } else {
                                sparse_lu.remove(&(j, k));
                            }
                        }
                    }
                }
            }
        }
        
        // Solve using the HashMap representation
        let mut x: Vector = (0..n).map(|i| b[banded.old(p[i])]).collect();
        
        // Forward substitution
        for i in 0..n {
            let mut sum = 0.0;
            let start_col = i.saturating_sub(bandwidth);
            for j in start_col..i {
                if let Some(&val) = sparse_lu.get(&(i, j)) {
                    sum += val * x[j];
                }
            }
            x[i] -= sum;
        }
        
        // Backward substitution
        for i in (0..n).rev() {
            let mut sum = 0.0;
            let end_col = (i + bandwidth + 1).min(n);
            for j in (i + 1)..end_col {
                if let Some(&val) = sparse_lu.get(&(i, j)) {
                    sum += val * x[j];
                }
            }
            x[i] -= sum;
            
            let diagonal = *sparse_lu.get(&(i, i)).unwrap_or(&0.0);
            if diagonal.abs() < ABS_TOL {
                return Err(SquareMatrixError::Singular);
            }
            x[i] /= diagonal;
        }
        
        Ok((0..n).map(|i| x[banded.map(i)]).collect())
    }
}

// fn forward_substitution(x: &Vector, a: &SparseSquareMatrix) -> Vector {
//     let mut y = Vector::zero(x.len());
//     for row in 0..x.len() {
//         let start = a.row_pointers[row];
//         let end = a.row_pointers[row + 1];
//         let mut sum = Scalar::default();
//         for i in start..end {
//             let col = a.col_indices[i];
//             if col < row {
//                 sum += a.values[i] * y[col];
//             }
//         }
//         y[row] = (x[row] - sum) / a[[row, row]];
//     }
//     y
// }

// fn backward_substitution(y: &Vector, a: &SparseSquareMatrix) -> Vector {
//     let mut x = Vector::zero(y.len());
//     for row in (0..y.len()).rev() {
//         let start = a.row_pointers[row];
//         let end = a.row_pointers[row + 1];
//         let mut sum = Scalar::default();
//         for i in start..end {
//             let col = a.col_indices[i];
//             if col > row {
//                 sum += a.values[i] * x[col];
//             }
//         }
//         x[row] = (y[row] - sum) / a[[row, row]];
//     }
//     x
// }

fn forward_substitution_sparse(x: &mut Vector, matrix: &[HashMap<usize, Scalar>]) {
    let n = matrix.len();
    
    for i in 0..n {
        let mut sum = Scalar::default();
        
        // Sum L[i,j] * x[j] for j < i
        for (&col, &value) in &matrix[i] {
            if col < i {
                sum += value * x[col];
            }
        }
        
        // Since L has 1's on diagonal: x[i] = (x[i] - sum) / 1.0
        x[i] -= sum;
    }
}

fn backward_substitution_sparse(x: &mut Vector, matrix: &[HashMap<usize, Scalar>]) {
    let n = matrix.len();
    
    for i in (0..n).rev() {
        let mut sum = Scalar::default();
        let mut diagonal = Scalar::default();
        
        // Find diagonal element and sum upper triangular elements
        for (&col, &value) in &matrix[i] {
            if col == i {
                diagonal = value;
            } else if col > i {
                sum += value * x[col];
            }
        }
        
        if diagonal.abs() < ABS_TOL {
            panic!("Zero diagonal element encountered during backward substitution at row {}", i);
        }
        
        x[i] = (x[i] - sum) / diagonal;
    }
}

impl Display for SparseSquareMatrix {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "\x1B[s")?;
        write!(f, "[[")?;
        (0..self.len).try_for_each(|row| {
            (0..self.len).try_for_each(|col| write_tensor_rank_0(f, &self[[row, col]]))?;
            if row + 1 < self.len {
                writeln!(f, "\x1B[2D],")?;
                write!(f, "\x1B[u")?;
                write!(f, "\x1B[{}B [", row + 1)?;
            }
            Ok(())
        })?;
        write!(f, "\x1B[2D]]")
    }
}

impl Index<[usize; 2]> for SparseSquareMatrix {
    type Output = Scalar;
    fn index(&self, [row, col]: [usize; 2]) -> &Self::Output {
        for i in self.row_pointers[row]..self.row_pointers[row + 1] {
            if self.col_indices[i] == col {
                return &self.values[i];
            }
            if self.col_indices[i] > col {
                break; // Assuming sorted columns
            }
        }
        &0.0
    }
}

impl Mul<&Self> for SparseSquareMatrix {
    type Output = Self;
    fn mul(self, other: &Self) -> Self::Output {
        assert!(
            self.len == other.len,
            "Matrix dimensions do not match for multiplication"
        );

        let mut result_values = Vec::new();
        let mut result_col_indices = Vec::new();
        let mut result_row_pointers = vec![0; self.len + 1];

        for row in 0..self.len {
            let mut row_result = std::collections::HashMap::new();

            let start_a = self.row_pointers[row];
            let end_a = self.row_pointers[row + 1];

            for i in start_a..end_a {
                let col_a = self.col_indices[i];
                let value_a = self.values[i];

                let start_b = other.row_pointers[col_a];
                let end_b = other.row_pointers[col_a + 1];

                for j in start_b..end_b {
                    let col_b = other.col_indices[j];
                    let value_b = other.values[j];

                    let entry = row_result.entry(col_b).or_insert(Scalar::default());
                    *entry += value_a * value_b;
                }
            }

            // Convert the row_result HashMap into CSR format for the result matrix.
            let mut sorted_row_result: Vec<(usize, Scalar)> = row_result.into_iter().collect();
            sorted_row_result.sort_by_key(|&(col, _)| col); // Ensure column indices are sorted.

            // for (&col, &value) in row_result.iter() {
            //     if value != Scalar::default() {
            //         result_values.push(value);
            //         result_col_indices.push(col);
            //     }
            // }
            for (col, value) in sorted_row_result {
                if value != Scalar::default() {
                    result_values.push(value);
                    result_col_indices.push(col);
                }
            }

            result_row_pointers[row + 1] = result_values.len();
        }

        Self {
            len: self.len,
            values: result_values,
            col_indices: result_col_indices,
            row_pointers: result_row_pointers,
        }
    }
}
