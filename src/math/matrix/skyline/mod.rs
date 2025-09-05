use crate::math::Scalar;

/// A skyline matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct SkylineMatrix {
    values: Vec<Scalar>,
    col_heights: Vec<usize>, // Height of each column (from first nonzero to diagonal)
    row_heights: Vec<usize>, // Height of each row (from first nonzero to diagonal)
    col_offsets: Vec<usize>, // Starting position of each column in values array
    row_offsets: Vec<usize>, // Starting position of each row in row_values array
    row_values: Vec<Scalar>, // Storage for lower triangle (row-wise)
    size: usize,
}

//
// Would need block.nodal_stiffnesses() to directly fill and return the skyline matrix in order to retain memory efficiency.
// Could have it return an enum or use two functions if want to keep both options (such as for all your tests).
// And then the solver parts will just call different impls for the two different options.
// The part in fem/block that builds the skyline matrix for the stiffness will also have to return information about the tangent skyline matrix.
// So that it can be built correctly in the solver. Could handle it in fem/block but that would assume something about how it is being solved.
// For example the separate range/null space method will use it differently.
//

impl SkylineMatrix {
    pub fn new(size: usize, nonzero_pattern: &[(usize, usize)]) -> Self {
        // Calculate column heights (upper triangle + diagonal, column-wise storage)
        let mut col_heights = vec![0; size];
        // Calculate row heights (lower triangle, row-wise storage)
        let mut row_heights = vec![0; size];

        for &(i, j) in nonzero_pattern {
            if i <= j {
                // Upper triangle or diagonal - store in column-wise format
                let first_nonzero_row = j + 1 - col_heights[j].max(1);
                if i < first_nonzero_row || col_heights[j] == 0 {
                    col_heights[j] = j - i + 1;
                }
            } else {
                // Lower triangle - store in row-wise format
                let first_nonzero_col = i + 1 - row_heights[i].max(1);
                if j < first_nonzero_col || row_heights[i] == 0 {
                    row_heights[i] = i - j + 1;
                }
            }
        }

        // Calculate column offsets (for upper triangle)
        let mut col_offsets = vec![0; size + 1];
        for j in 1..=size {
            col_offsets[j] = col_offsets[j - 1] + col_heights[j - 1];
        }

        // Calculate row offsets (for lower triangle)
        let mut row_offsets = vec![0; size + 1];
        for i in 1..=size {
            row_offsets[i] = row_offsets[i - 1] + row_heights[i - 1];
        }

        Self {
            values: vec![0.0; col_offsets[size]], // Upper triangle + diagonal
            row_values: vec![0.0; row_offsets[size]], // Lower triangle
            col_heights,
            row_heights,
            col_offsets,
            row_offsets,
            size,
        }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i <= j {
            // Upper triangle or diagonal - use column storage
            if self.col_heights[j] == 0 {
                return 0.0;
            }

            let first_row = j + 1 - self.col_heights[j];
            if i < first_row {
                0.0
            } else {
                self.values[self.col_offsets[j] + (i - first_row)]
            }
        } else {
            // Lower triangle - use row storage
            if self.row_heights[i] == 0 {
                return 0.0;
            }

            let first_col = i + 1 - self.row_heights[i];
            if j < first_col {
                0.0
            } else {
                self.row_values[self.row_offsets[i] + (j - first_col)]
            }
        }
    }

    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        if i <= j {
            // Upper triangle or diagonal - use column storage
            if self.col_heights[j] > 0 {
                let first_row = j + 1 - self.col_heights[j];
                if i >= first_row {
                    self.values[self.col_offsets[j] + (i - first_row)] = value;
                }
            }
        } else {
            // Lower triangle - use row storage
            if self.row_heights[i] > 0 {
                let first_col = i + 1 - self.row_heights[i];
                if j >= first_col {
                    self.row_values[self.row_offsets[i] + (j - first_col)] = value;
                }
            }
        }
    }

    // // Helper methods for iteration
    // pub fn upper_triangle_iter(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
    //     (0..self.size).flat_map(move |j| {
    //         if self.col_heights[j] > 0 {
    //             let first_row = j + 1 - self.col_heights[j];
    //             (first_row..=j).map(move |i| {
    //                 let value = self.values[self.col_offsets[j] + (i - first_row)];
    //                 (i, j, value)
    //             })
    //         } else {
    //             std::iter::empty().collect::<Vec<_>>().into_iter()
    //         }
    //     })
    // }

    // pub fn lower_triangle_iter(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
    //     (0..self.size).flat_map(move |i| {
    //         if self.row_heights[i] > 0 {
    //             let first_col = i + 1 - self.row_heights[i];
    //             (first_col..i).map(move |j| { // Note: j < i for lower triangle
    //                 let value = self.row_values[self.row_offsets[i] + (j - first_col)];
    //                 (i, j, value)
    //             })
    //         } else {
    //             std::iter::empty().collect::<Vec<_>>().into_iter()
    //         }
    //     })
    // }

    // Get skyline profile information
    pub fn get_column_height(&self, col: usize) -> usize {
        self.col_heights[col]
    }

    pub fn get_row_height(&self, row: usize) -> usize {
        self.row_heights[row]
    }

    // Memory usage
    pub fn memory_usage(&self) -> usize {
        self.values.len() + self.row_values.len()
    }

    pub fn total_stored_elements(&self) -> usize {
        self.col_heights.iter().sum::<usize>() + self.row_heights.iter().sum::<usize>()
    }
}
