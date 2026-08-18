#[cfg(test)]
mod test;

use crate::math::{Scalar, Tensor, Vector};

/// How near zero a pivot is allowed to come, once the matrix has been scaled so
/// that its own diagonal is one, before it is held off at that distance instead.
const FLOOR: Scalar = 1e-8;

/// How far the factorization is allowed to run away, on that same scaled matrix,
/// before it is not worth having at all.
///
/// Bounding the entries does not bound the pivots, which are sums of products of
/// them, and a factorization whose pivots have run to this is one that pivoting
/// would have prevented and that nothing here can repair.
const UNSTABLE: Scalar = 1e8;

/// How large an entry of the triangle is allowed to be, on the same scaled
/// matrix, before it is taken as growth and dropped.
///
/// Unpivoted elimination on an indefinite matrix has no bound on how large its
/// factor can become: a pivot small against what it divides throws out a large
/// entry, that entry feeds the corrections to every row after it, and the whole
/// factorization can run away by many orders of magnitude. A complete
/// factorization answers that by pivoting. An incomplete one, which is already
/// throwing entries away, can answer it by throwing these away too — an entry
/// this large is one the elimination could not compute stably, and keeping it
/// is worse than losing it.
const GROWTH: Scalar = 1e4;

/// An incomplete LDLᵀ factorization, kept to the sparsity it was given.
///
/// A complete factorization of a sparse matrix fills in: entries the matrix
/// held at zero become nonzero, and on a three-dimensional mesh there are far
/// more of them than of the entries the matrix had. Dropping every one of them
/// leaves a factor that costs what the matrix costs to store and to apply, and
/// that is no longer the matrix's factor — but it is near enough to stand in
/// for its inverse, which is all a preconditioner is asked to do.
///
/// Why LDLᵀ and not Cholesky. A tangent away from a minimum is indefinite, and
/// an indefinite matrix has no Cholesky factorization to be incomplete about —
/// the attempt ends at the square root of the first nonpositive pivot. Lifting
/// the diagonal until it survives is the usual answer, but the lift needed is
/// no small correction once the matrix is genuinely indefinite, and what gets
/// factorized is then not the matrix. Splitting the pivots out into a diagonal
/// of their own asks for no square root, so nothing breaks down and nothing has
/// to be shifted.
///
/// What is done with a negative pivot instead is to take its magnitude when the
/// factorization is applied — see [`Self::solve`]. Only the sign is discarded,
/// and only where the matrix has none to give a preconditioner.
pub struct CscIncompleteLdl {
    /// The strictly lower triangle by rows, columns ascending within each row.
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<Scalar>,
    /// The pivots, held apart from the triangle they came out of.
    pivots: Vec<Scalar>,
    /// What each row of the matrix was scaled by before any of this began.
    ///
    /// The factorization is of the scaled matrix, so the scaling is applied
    /// either side of every solve to put the answer back in the terms it was
    /// asked in.
    scaling: Vec<Scalar>,
    size: usize,
    /// How large the factor grew, and how many entries were dropped for growing.
    growth: (Scalar, usize),
}

impl CscIncompleteLdl {
    /// Gathers the lower triangle into rows and factorizes it.
    ///
    /// Positions repeated in the entries are summed, the caller being free to
    /// hand over a triangle assembled rather than one already merged.
    ///
    /// A row without a diagonal has no pivot, and a matrix missing one is not
    /// one this stands in for.
    pub fn new(
        size: usize,
        entries: impl IntoIterator<Item = (usize, usize, Scalar)>,
    ) -> Option<Self> {
        Self::with_fill(size, entries, 0, 0.0)
    }
    /// The same, keeping some of the fill rather than none of it.
    ///
    /// `fill` is how many entries a row of the factor may keep beyond the ones
    /// the matrix itself holds there, and `threshold` is how small an entry has
    /// to be, against the row of the matrix it came from, to be dropped whatever
    /// the room left. Zero and zero keep exactly the matrix's own positions.
    ///
    /// Fill is what stands between the factorization and the matrix's real one,
    /// so this is the dial that says how near to buy at what price. What it
    /// costs is not only room: an entry dropped early is one whose corrections
    /// are never applied, so keeping more of them costs time in the
    /// factorization as well as in every application of it.
    pub fn with_fill(
        size: usize,
        entries: impl IntoIterator<Item = (usize, usize, Scalar)>,
        fill: usize,
        threshold: Scalar,
    ) -> Option<Self> {
        let mut gathered: Vec<(usize, usize, Scalar)> = entries
            .into_iter()
            .filter(|&(row, column, _)| row >= column)
            .collect();
        gathered.sort_unstable_by_key(|&(row, column, _)| (row, column));
        let mut matrix_ptr = vec![0_usize; size + 1];
        let mut matrix_col = Vec::with_capacity(gathered.len());
        let mut matrix_val: Vec<Scalar> = Vec::with_capacity(gathered.len());
        let mut pivots = vec![Scalar::NAN; size];
        let mut last = (usize::MAX, usize::MAX);
        gathered.into_iter().for_each(|(row, column, value)| {
            if row == column {
                if pivots[row].is_nan() {
                    pivots[row] = value
                } else {
                    pivots[row] += value
                }
            } else if (row, column) == last {
                *matrix_val.last_mut().unwrap() += value
            } else {
                last = (row, column);
                matrix_col.push(column);
                matrix_val.push(value);
                matrix_ptr[row + 1] += 1
            }
        });
        if pivots.iter().any(|pivot| pivot.is_nan()) {
            return None;
        }
        (0..size).for_each(|row| matrix_ptr[row + 1] += matrix_ptr[row]);
        //
        // The matrix is scaled so that every diagonal entry is one before any of
        // it is eliminated. Unpivoted elimination is unstable in proportion to
        // how unlike each other the rows are, and a tangent assembled from a
        // mesh has rows differing by whatever the mesh and the material differ
        // by; putting them all on the same footing first is the cheapest thing
        // that answers it, and it costs the factorization nothing else.
        //
        let scaling: Vec<Scalar> = pivots
            .iter()
            .map(|pivot| match pivot.abs() {
                0.0 => 1.0,
                magnitude => magnitude.sqrt().recip(),
            })
            .collect();
        (0..size).for_each(|row| {
            pivots[row] *= scaling[row] * scaling[row];
            (matrix_ptr[row]..matrix_ptr[row + 1])
                .for_each(|k| matrix_val[k] *= scaling[row] * scaling[matrix_col[k]])
        });
        let factorization = Self::factorize(
            size, matrix_ptr, matrix_col, matrix_val, pivots, scaling, fill, threshold,
        );
        //
        // A factorization that ran away is refused rather than handed back. What
        // it produces is not a bad preconditioner but a meaningless one: the
        // walk measures its residual through the preconditioner, so a
        // preconditioner this far from the matrix reports a residual that is not
        // the residual, and stops early on an answer that is not an answer.
        // Saying there is none leaves the caller free to fall back on something
        // that is merely weak.
        //
        (factorization.growth.0 <= UNSTABLE).then_some(factorization)
    }
    /// Builds the factor a row at a time.
    ///
    /// A row of the factor is the matrix's row less what every earlier row
    /// accounts for, and what an earlier row accounts for reaches the columns
    /// its own entries reach. So the earlier rows are met not by looking them up
    /// but by having each finished row file itself under the columns it holds,
    /// and the row being built is then assembled by scattering those columns
    /// into it as they come due, in the order the columns come.
    ///
    /// The order matters: a correction can put an entry where the matrix had
    /// none, and that entry is itself due for correction later in the same row.
    /// Taking the columns in ascending order is what makes every correction
    /// arrive before the entry it lands on is finished, and what makes fill a
    /// thing that can be noticed at all — a position not held by the matrix is
    /// exactly one that arrived this way.
    #[allow(clippy::too_many_arguments)]
    fn factorize(
        size: usize,
        matrix_ptr: Vec<usize>,
        matrix_col: Vec<usize>,
        matrix_val: Vec<Scalar>,
        diagonal: Vec<Scalar>,
        scaling: Vec<Scalar>,
        fill: usize,
        threshold: Scalar,
    ) -> Self {
        //
        // A pivot at zero is the one thing that still has no answer, division by
        // it being what the next rows would do. The matrix arrives scaled to a
        // diagonal of one, so how near zero is too near is a plain number rather
        // than something measured against the matrix.
        //
        let floor = FLOOR;
        let mut grew = (0.0 as Scalar, 0);
        let mut pivots = vec![0.0; size];
        let mut row_ptr = vec![0_usize; size + 1];
        let mut col_idx = Vec::with_capacity(matrix_col.len());
        let mut values = Vec::with_capacity(matrix_val.len());
        let mut columns: Vec<Vec<(usize, Scalar)>> = vec![Vec::new(); size];
        let mut work = vec![0.0; size];
        let mut reached = Vec::new();
        let mut held = vec![false; size];
        let mut kept = vec![false; size];
        let mut pending = std::collections::BinaryHeap::new();
        let mut row_entries: Vec<(usize, Scalar)> = Vec::new();
        for row in 0..size {
            let (start, stop) = (matrix_ptr[row], matrix_ptr[row + 1]);
            let scale = threshold
                * (diagonal[row] * diagonal[row]
                    + (start..stop)
                        .map(|k| matrix_val[k] * matrix_val[k])
                        .sum::<Scalar>())
                .sqrt();
            (start..stop).for_each(|k| {
                let column = matrix_col[k];
                work[column] = matrix_val[k];
                held[column] = true;
                reached.push(column);
                pending.push(std::cmp::Reverse(column))
            });
            row_entries.clear();
            while let Some(std::cmp::Reverse(column)) = pending.pop() {
                let entry = work[column] / pivots[column];
                //
                // A position the matrix holds is kept whatever its size, so that
                // asking for no fill asks for the matrix's own pattern exactly
                // and for nothing to depend on how the values came out. Where
                // there is no room for fill at all, an arrival is turned away
                // here rather than after it has spread its own corrections
                // through the row, which is what makes that case exactly the
                // factorization restricted to the matrix's pattern.
                //
                if !held[column] && (fill == 0 || entry.abs() * pivots[column].abs().sqrt() < scale)
                {
                    work[column] = 0.0;
                    continue;
                }
                //
                // An entry this large is one the elimination could not compute
                // stably. It is dropped whether or not the matrix held that
                // position, which is the one place the matrix's own pattern is
                // not honoured — and the place where honouring it would mean
                // building a preconditioner out of numbers that mean nothing.
                //
                grew = (grew.0.max(entry.abs()), grew.1);
                if entry.abs() > GROWTH {
                    grew = (grew.0, grew.1 + 1);
                    work[column] = 0.0;
                    continue;
                }
                work[column] = entry;
                row_entries.push((column, entry));
                columns[column].iter().for_each(|&(later, value)| {
                    if !held[later] && !kept[later] {
                        kept[later] = true;
                        work[later] = 0.0;
                        reached.push(later);
                        pending.push(std::cmp::Reverse(later))
                    }
                    work[later] -= entry * pivots[column] * value
                })
            }
            //
            // What the matrix held is kept outright; the rest competes for the
            // room left, largest first, measured in what it contributes to the
            // factor rather than in the bare entry.
            //
            if fill > 0 {
                let mut arrived: Vec<(usize, Scalar)> = row_entries
                    .iter()
                    .filter(|&&(column, _)| !held[column])
                    .copied()
                    .collect();
                if arrived.len() > fill {
                    arrived.sort_unstable_by(|a, b| {
                        (b.1 * b.1 * pivots[b.0].abs())
                            .partial_cmp(&(a.1 * a.1 * pivots[a.0].abs()))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    arrived.truncate(fill);
                    row_entries.retain(|&(column, _)| held[column]);
                    row_entries.append(&mut arrived)
                }
            }
            row_entries.sort_unstable_by_key(|&(column, _)| column);
            let pivot = diagonal[row]
                - row_entries
                    .iter()
                    .map(|&(column, entry)| entry * entry * pivots[column])
                    .sum::<Scalar>();
            pivots[row] = if !pivot.is_finite() {
                floor
            } else if pivot.abs() < floor {
                floor.copysign(pivot)
            } else {
                pivot
            };
            grew = (grew.0.max(pivots[row].abs()), grew.1);
            row_entries.iter().for_each(|&(column, entry)| {
                col_idx.push(column);
                values.push(entry);
                columns[column].push((row, entry))
            });
            row_ptr[row + 1] = col_idx.len();
            reached.drain(..).for_each(|column| {
                work[column] = 0.0;
                held[column] = false;
                kept[column] = false
            })
        }
        Self {
            row_ptr,
            col_idx,
            values,
            pivots,
            scaling,
            size,
            growth: grew,
        }
    }
    /// How many pivots came out negative.
    ///
    /// What the matrix could not be told about itself before it was factorized:
    /// this is zero exactly when the matrix is positive definite.
    pub fn negative_pivots(&self) -> usize {
        self.pivots.iter().filter(|&&pivot| pivot < 0.0).count()
    }
    /// What the factorization makes of a vector, standing in for the inverse.
    ///
    /// The pivots are taken by magnitude, so what is solved against is not the
    /// matrix factorized but the nearest positive definite rearrangement of it —
    /// the same factor, the same directions, every negative curvature turned
    /// positive. A preconditioner has to be positive definite for the walk to
    /// hold its footing, and this is what an indefinite matrix has to offer that
    /// is: on the half of the spectrum that was already positive it changes
    /// nothing, and on the half that was not it maps the eigenvalues onto the
    /// same cluster as their opposites rather than leaving them straddling zero.
    pub fn solve(&self, right_hand_side: &Vector) -> Vector {
        let mut solution = Vector::zero(self.size);
        self.solve_into(right_hand_side, &mut solution);
        solution
    }
    /// The same, into a vector already standing.
    pub fn solve_into(&self, right_hand_side: &Vector, solution: &mut Vector) {
        let forward = &mut vec![0.0; self.size];
        let scaled: Vec<Scalar> = right_hand_side
            .iter()
            .zip(self.scaling.iter())
            .map(|(entry, scale)| entry * scale)
            .collect();
        (0..self.size).for_each(|row| {
            forward[row] = scaled[row]
                - (self.row_ptr[row]..self.row_ptr[row + 1])
                    .map(|k| self.values[k] * forward[self.col_idx[k]])
                    .sum::<Scalar>()
        });
        forward
            .iter_mut()
            .zip(self.pivots.iter())
            .for_each(|(entry, pivot)| *entry /= pivot.abs());
        //
        // The transpose is walked by rows as well, each finished entry paying
        // back into the columns its row reaches rather than being gathered from
        // a row of the transpose that is nowhere stored.
        //
        (0..self.size).rev().for_each(|row| {
            let value = forward[row];
            (self.row_ptr[row]..self.row_ptr[row + 1])
                .for_each(|k| forward[self.col_idx[k]] -= self.values[k] * value)
        });
        solution
            .iter_mut()
            .zip(forward.iter().zip(self.scaling.iter()))
            .for_each(|(entry, (from, scale))| *entry = from * scale)
    }
    pub fn size(&self) -> usize {
        self.size
    }
    /// The entry of the unit lower triangle at a position, the diagonal being
    /// the one it does not store.
    #[cfg(test)]
    fn entry(&self, row: usize, column: usize) -> Scalar {
        if row == column {
            1.0
        } else {
            (self.row_ptr[row]..self.row_ptr[row + 1])
                .find(|&k| self.col_idx[k] == column)
                .map_or(0.0, |k| self.values[k])
        }
    }
    #[cfg(test)]
    fn pivot(&self, row: usize) -> Scalar {
        self.pivots[row]
    }
    #[cfg(test)]
    fn scale(&self, row: usize) -> Scalar {
        self.scaling[row]
    }
}
