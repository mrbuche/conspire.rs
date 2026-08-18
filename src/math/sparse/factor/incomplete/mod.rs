#[cfg(test)]
mod test;

use crate::math::{Scalar, Vector};

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

/// How much a kept fill entry is allowed to amplify the row of the inverse
/// factor it depends on, before it is dropped for that instead of for its own
/// size.
///
/// `GROWTH` bounds the entries of the factor itself, but the factor is not
/// what a solve touches — its inverse is, one triangular pass at a time, and
/// an entry can be individually unremarkable while still sitting at the end
/// of a dependency chain whose own inverse has already grown large. Push a
/// correction through that chain and what comes out is *entry × however much
/// the chain already amplifies*, not just the entry — an amplification
/// `GROWTH` alone has no way to see, since it only ever looks at one row at a
/// time. Bounding the entries of `L` does not bound `L⁻¹`; this bounds it
/// directly, weighing a candidate entry against `rho`, a running bound on the
/// row-sum norm of the inverse factor at the dependency it came from — which
/// is exactly what a triangular solve multiplies a dropped correction by.
/// Rediscovered the hard way: a factor whose every pivot passed `UNSTABLE`
/// and whose every entry passed `GROWTH` still made a walk's own
/// positive-definiteness check come out negative, on the real deformed
/// tangent this exists for — see `PIVOTING.md`.
const AMPLIFICATION: Scalar = 1e10;

/// How far the worst inverse row-sum bound is allowed to grow before the
/// factorization is refused outright, the same way `UNSTABLE` already refuses
/// one that grew in its own entries.
const INVERSE_UNSTABLE: Scalar = 1e14;

/// How small the natural next candidate's own diagonal has to have fallen,
/// on the scaled matrix, before it is passed over for the best one available.
///
/// This is judged against the candidate alone, not against whatever else is
/// on offer: comparing it to the current largest would reorder every step
/// some other index happens to edge the natural one out by, on a matrix
/// where every diagonal is merely close rather than tied — and reordering is
/// not free even where it is safe, since it is what disturbs whatever
/// locality the matrix's own numbering gave the fill this keeps. What
/// actually threatens the factorization is a pivot that has fallen toward
/// zero, not one that is merely no longer the biggest.
///
/// Measured against the real hyperelastic block this was built for (strain
/// 13, 3993 free variables — see `PIVOTING.md`): the window between "leaves
/// the healthy reference-configuration tangent alone" and "actually rescues
/// the deformed one from refusal" is narrow and not monotonic in between —
/// 4.5e-2 and 4.9e-2 both still refuse where 4e-2 and 5e-2 do not. 4e-2 is
/// the smallest value that stopped the deformed tangent's factorization
/// being refused in that measurement, chosen over 5e-2 for being the more
/// conservative of the two working points, and confirmed to leave the
/// reference-configuration tangent's elimination order untouched — its
/// solve there is byte-identical to elimination forced into the matrix's own
/// order.
const SAFE: Scalar = 4e-2;

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
///
/// The factorization is of `P A Pᵀ`, not of `A` itself. Unpivoted elimination on
/// an indefinite matrix has no bound on how far it can run away — a pivot small
/// against what it divides throws out a large entry, and that entry poisons
/// every correction after it. `P` is chosen during elimination, one index at a
/// time, as whichever remaining index currently has the largest-magnitude
/// diagonal entry of the Schur complement — the largest pivot on offer is the
/// one least likely to be small against what it divides. Tracking that diagonal
/// as elimination proceeds, rather than only discovering it the moment an index
/// is eliminated, is what makes the choice possible at all: the moment a column
/// finalizes, its effect on every remaining index it touches is folded into
/// that index's running diagonal immediately, so comparing candidates is always
/// a lookup and never a recomputation.
pub struct CscIncompleteLdl {
    /// The strictly lower triangle by rows of elimination order, columns
    /// ascending within each row.
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<Scalar>,
    /// The pivots, in elimination order, held apart from the triangle they came
    /// out of.
    pivots: Vec<Scalar>,
    /// What each row of the matrix was scaled by before any of this began, kept
    /// in the matrix's own indices.
    ///
    /// The factorization is of the scaled matrix, so the scaling is applied
    /// either side of every solve to put the answer back in the terms it was
    /// asked in.
    scaling: Vec<Scalar>,
    /// The permutation elimination chose: `permutation[step]` is the matrix's
    /// own index of whatever was eliminated at that step.
    permutation: Vec<usize>,
    /// The inverse of `permutation`: `position[index]` is the step at which the
    /// matrix's own `index` was eliminated. `solve` never needs to go from an
    /// index to its step, only ever the other way, so this is kept only for
    /// what reads a factorization by the matrix's own indices — today that is
    /// only the tests.
    #[cfg_attr(not(test), allow(dead_code))]
    position: Vec<usize>,
    size: usize,
    /// How large the factor grew, and how many entries were dropped for growing.
    growth: (Scalar, usize),
    /// How large the worst bound on a row of the inverse factor grew, and how
    /// many entries were dropped for amplifying it too far — see
    /// [`AMPLIFICATION`].
    inverse_growth: (Scalar, usize),
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
        let mut off_diagonal: Vec<(usize, usize, Scalar)> = Vec::with_capacity(gathered.len());
        let mut diagonal = vec![Scalar::NAN; size];
        let mut last = (usize::MAX, usize::MAX);
        gathered.into_iter().for_each(|(row, column, value)| {
            if row == column {
                if diagonal[row].is_nan() {
                    diagonal[row] = value
                } else {
                    diagonal[row] += value
                }
            } else if (row, column) == last {
                off_diagonal.last_mut().unwrap().2 += value
            } else {
                last = (row, column);
                off_diagonal.push((row, column, value))
            }
        });
        if diagonal.iter().any(|entry| entry.is_nan()) {
            return None;
        }
        //
        // The matrix is scaled so that every diagonal entry is one before any of
        // it is eliminated. Unpivoted elimination is unstable in proportion to
        // how unlike each other the rows are, and a tangent assembled from a
        // mesh has rows differing by whatever the mesh and the material differ
        // by; putting them all on the same footing first is the cheapest thing
        // that answers it, and it costs the factorization nothing else.
        //
        let scaling: Vec<Scalar> = diagonal
            .iter()
            .map(|entry| match entry.abs() {
                0.0 => 1.0,
                magnitude => magnitude.sqrt().recip(),
            })
            .collect();
        diagonal
            .iter_mut()
            .zip(scaling.iter())
            .for_each(|(entry, scale)| *entry *= scale * scale);
        let mut neighbors: Vec<Vec<(usize, Scalar)>> = vec![Vec::new(); size];
        off_diagonal.into_iter().for_each(|(row, column, value)| {
            let scaled = value * scaling[row] * scaling[column];
            neighbors[row].push((column, scaled));
            neighbors[column].push((row, scaled))
        });
        let factorization = Self::factorize(size, neighbors, diagonal, scaling, fill, threshold);
        //
        // A factorization that ran away is refused rather than handed back. What
        // it produces is not a bad preconditioner but a meaningless one: the
        // walk measures its residual through the preconditioner, so a
        // preconditioner this far from the matrix reports a residual that is not
        // the residual, and stops early on an answer that is not an answer.
        // Saying there is none leaves the caller free to fall back on something
        // that is merely weak.
        //
        (factorization.growth.0 <= UNSTABLE && factorization.inverse_growth.0 <= INVERSE_UNSTABLE)
            .then_some(factorization)
    }
    /// Builds the factor one pivot at a time, choosing at every step whichever
    /// remaining index currently carries the largest-magnitude diagonal entry of
    /// the Schur complement.
    ///
    /// `diag` is that running diagonal, correct at every step for every index
    /// not yet eliminated: the moment a column finalizes, its effect on every
    /// remaining index it still touches is subtracted from `diag` immediately,
    /// using exactly the entries just computed to finish that column — nothing
    /// is ever recomputed to make a comparison. Picking the largest available
    /// diagonal entry is what stands in here for the threshold test proper
    /// Bunch-Kaufman pivoting runs against a candidate 2×2 block; there is no
    /// 2×2 block yet, so there is nothing to test against, and the closest thing
    /// this can still refuse is a step where nothing remaining rises above
    /// `FLOOR` — handled the same way a small pivot always has been, by holding
    /// it off at that distance rather than failing the whole factorization for
    /// one bad step.
    ///
    /// A column, once chosen, is built the same way a row of the old row-order
    /// factorization was: its own entries are scattered, and corrections are
    /// pulled in from every already-eliminated index it reaches, in the order
    /// those indices were eliminated, via `columns` — but where the old
    /// factorization only ever looked backward, at the indices before it in a
    /// fixed order, this one has no fixed order to look backward through, so
    /// `columns` is generalized: it holds, for a finalized index, not only what
    /// depends on it (used the same way as before) but also what it still
    /// depends on nothing, which is to say every not-yet-eliminated index it
    /// reaches — the two directions of a symmetric matrix restricted to what
    /// elimination order has actually resolved.
    #[allow(clippy::too_many_arguments)]
    fn factorize(
        size: usize,
        neighbors: Vec<Vec<(usize, Scalar)>>,
        diagonal: Vec<Scalar>,
        scaling: Vec<Scalar>,
        fill: usize,
        threshold: Scalar,
    ) -> Self {
        let floor = FLOOR;
        let mut grew = (0.0 as Scalar, 0);
        let mut inverse_grew = (0.0 as Scalar, 0);
        //
        // `rho[step]` bounds the row-sum norm of row `step` of the inverse
        // factor, `‖L⁻¹‖` restricted to that row. For a unit lower triangular
        // matrix, row `i` of `L⁻¹` satisfies `(L⁻¹)ᵢⱼ = δᵢⱼ - Σₖ Lᵢₖ(L⁻¹)ₖⱼ` for
        // `k < i`, so its row-sum norm is bounded by `1 + Σₖ |Lᵢₖ| · ρₖ` —
        // exactly the entries and dependencies already in hand once a row is
        // built, so this costs nothing beyond what factoring already touches.
        //
        let mut rho = vec![0.0; size];
        let mut diag = diagonal.clone();
        let mut eliminated = vec![false; size];
        let mut position = vec![0_usize; size];
        let mut permutation = Vec::with_capacity(size);
        let mut pivots: Vec<Scalar> = Vec::with_capacity(size);
        let mut row_ptr = vec![0_usize; size + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        let mut columns: Vec<Vec<(usize, Scalar)>> = vec![Vec::new(); size];
        let mut work = vec![0.0; size];
        let mut held = vec![false; size];
        let mut kept = vec![false; size];
        let mut reached = Vec::new();
        let mut pending = std::collections::BinaryHeap::new();
        let mut dependencies: Vec<(usize, Scalar)> = Vec::new();
        let mut dependents: Vec<(usize, Scalar)> = Vec::new();
        for step in 0..size {
            //
            // The next index in the order the matrix itself came in is used
            // outright unless its own diagonal has fallen well behind the best
            // one still on offer — reordering is not free even where it is
            // safe, since it is what disturbs whatever locality the matrix's
            // own numbering held for the fill this keeps, so it is spent only
            // where the natural candidate could not stand behind a pivot
            // threshold on its own.
            //
            let candidate = (0..size).find(|&index| !eliminated[index]).unwrap();
            let pivot_index = if diag[candidate].abs() >= SAFE {
                candidate
            } else {
                let mut largest_index = candidate;
                let mut largest = -1.0 as Scalar;
                (0..size).for_each(|index| {
                    if !eliminated[index] && diag[index].abs() > largest {
                        largest = diag[index].abs();
                        largest_index = index
                    }
                });
                largest_index
            };
            let scale = threshold
                * (diagonal[pivot_index] * diagonal[pivot_index]
                    + neighbors[pivot_index]
                        .iter()
                        .map(|&(_, value)| value * value)
                        .sum::<Scalar>())
                .sqrt();
            neighbors[pivot_index].iter().for_each(|&(j, value)| {
                work[j] = value;
                held[j] = true;
                reached.push(j);
                if eliminated[j] {
                    pending.push(std::cmp::Reverse((position[j], j)))
                }
            });
            dependencies.clear();
            while let Some(std::cmp::Reverse((_, column))) = pending.pop() {
                let entry = work[column] / pivots[position[column]];
                //
                // A position the matrix holds is kept whatever its size, so that
                // asking for no fill asks for the matrix's own pattern exactly.
                //
                if !held[column]
                    && (fill == 0 || entry.abs() * pivots[position[column]].abs().sqrt() < scale)
                {
                    work[column] = 0.0;
                    continue;
                }
                grew = (grew.0.max(entry.abs()), grew.1);
                if entry.abs() > GROWTH {
                    grew = (grew.0, grew.1 + 1);
                    work[column] = 0.0;
                    continue;
                }
                let amplified = entry.abs() * rho[position[column]];
                inverse_grew = (inverse_grew.0.max(amplified), inverse_grew.1);
                if !held[column] && amplified > AMPLIFICATION {
                    inverse_grew = (inverse_grew.0, inverse_grew.1 + 1);
                    work[column] = 0.0;
                    continue;
                }
                work[column] = entry;
                dependencies.push((column, entry));
                columns[column].iter().for_each(|&(later, value)| {
                    if !held[later] && !kept[later] {
                        kept[later] = true;
                        work[later] = 0.0;
                        reached.push(later);
                        if eliminated[later] {
                            pending.push(std::cmp::Reverse((position[later], later)))
                        }
                    }
                    work[later] -= entry * pivots[position[column]] * value
                })
            }
            //
            // What the matrix held is kept outright; the rest competes for the
            // room left, largest first, measured in what it contributes to the
            // factor rather than in the bare entry.
            //
            if fill > 0 {
                let mut arrived: Vec<(usize, Scalar)> = dependencies
                    .iter()
                    .filter(|&&(column, _)| !held[column])
                    .copied()
                    .collect();
                if arrived.len() > fill {
                    arrived.sort_unstable_by(|a, b| {
                        (b.1 * b.1 * pivots[position[b.0]].abs())
                            .partial_cmp(&(a.1 * a.1 * pivots[position[a.0]].abs()))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    arrived.truncate(fill);
                    dependencies.retain(|&(column, _)| held[column]);
                    dependencies.append(&mut arrived)
                }
            }
            dependencies.sort_unstable_by_key(|&(column, _)| position[column]);
            rho[step] = 1.0
                + dependencies
                    .iter()
                    .map(|&(column, entry)| entry.abs() * rho[position[column]])
                    .sum::<Scalar>();
            inverse_grew = (inverse_grew.0.max(rho[step]), inverse_grew.1);
            let pivot = diag[pivot_index];
            let pivot = if !pivot.is_finite() {
                floor
            } else if pivot.abs() < floor {
                floor.copysign(pivot)
            } else {
                pivot
            };
            grew = (grew.0.max(pivot.abs()), grew.1);
            dependencies.iter().for_each(|&(column, entry)| {
                col_idx.push(position[column]);
                values.push(entry)
            });
            row_ptr[step + 1] = col_idx.len();
            //
            // The other direction: not-yet-eliminated indices this pivot still
            // reaches are divided by its own pivot rather than an earlier one,
            // recorded for the future in `columns`, and their share of what this
            // pivot took out of the Schur complement is subtracted from `diag`
            // immediately — the only reason a later step's comparison is ever a
            // lookup rather than a recomputation.
            //
            dependents.clear();
            reached
                .iter()
                .copied()
                .filter(|&index| !eliminated[index])
                .for_each(|index| {
                    let entry = work[index] / pivot;
                    if !held[index] && (fill == 0 || entry.abs() * pivot.abs().sqrt() < scale) {
                        return;
                    }
                    if entry.abs() > GROWTH {
                        grew = (grew.0.max(entry.abs()), grew.1 + 1);
                        return;
                    }
                    grew = (grew.0.max(entry.abs()), grew.1);
                    dependents.push((index, entry))
                });
            if fill > 0 {
                let mut arrived: Vec<(usize, Scalar)> = dependents
                    .iter()
                    .filter(|&&(index, _)| !held[index])
                    .copied()
                    .collect();
                if arrived.len() > fill {
                    arrived.sort_unstable_by(|a, b| {
                        (b.1 * b.1)
                            .partial_cmp(&(a.1 * a.1))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    arrived.truncate(fill);
                    dependents.retain(|&(index, _)| held[index]);
                    dependents.append(&mut arrived)
                }
            }
            dependents
                .iter()
                .for_each(|&(index, entry)| diag[index] -= entry * entry * pivot);
            columns[pivot_index] = std::mem::take(&mut dependents);
            position[pivot_index] = step;
            eliminated[pivot_index] = true;
            permutation.push(pivot_index);
            pivots.push(pivot);
            reached.drain(..).for_each(|index| {
                work[index] = 0.0;
                held[index] = false;
                kept[index] = false
            })
        }
        Self {
            row_ptr,
            col_idx,
            values,
            pivots,
            scaling,
            permutation,
            position,
            size,
            growth: grew,
            inverse_growth: inverse_grew,
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
        let scaled: Vec<Scalar> = self
            .permutation
            .iter()
            .map(|&index| right_hand_side[index] * self.scaling[index])
            .collect();
        (0..self.size).for_each(|step| {
            forward[step] = scaled[step]
                - (self.row_ptr[step]..self.row_ptr[step + 1])
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
        (0..self.size).rev().for_each(|step| {
            let value = forward[step];
            (self.row_ptr[step]..self.row_ptr[step + 1])
                .for_each(|k| forward[self.col_idx[k]] -= self.values[k] * value)
        });
        self.permutation
            .iter()
            .zip(forward.iter())
            .for_each(|(&index, &from)| solution[index] = from * self.scaling[index])
    }
    pub fn size(&self) -> usize {
        self.size
    }
    /// The entry of the unit lower triangle at a position, the diagonal being
    /// the one it does not store — both positions taken in the matrix's own
    /// indices, elimination order being an internal matter.
    #[cfg(test)]
    fn entry(&self, row: usize, column: usize) -> Scalar {
        if row == column {
            1.0
        } else if self.position[column] >= self.position[row] {
            0.0
        } else {
            let step = self.position[row];
            (self.row_ptr[step]..self.row_ptr[step + 1])
                .find(|&k| self.col_idx[k] == self.position[column])
                .map_or(0.0, |k| self.values[k])
        }
    }
    #[cfg(test)]
    fn pivot(&self, row: usize) -> Scalar {
        self.pivots[self.position[row]]
    }
    #[cfg(test)]
    fn scale(&self, row: usize) -> Scalar {
        self.scaling[row]
    }
    #[cfg(test)]
    fn growth(&self) -> (Scalar, usize) {
        self.growth
    }
    #[cfg(test)]
    fn inverse_growth(&self) -> (Scalar, usize) {
        self.inverse_growth
    }
}
