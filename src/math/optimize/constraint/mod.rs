#[cfg(test)]
mod test;

use crate::math::{Matrix, Scalar, Solution, Vector, sparse::CscMatrix};
use std::collections::{BTreeMap, BTreeSet};

/// Why a solver cannot take a constraint.
pub(super) const UNSUPPORTED: &str =
    "This solver takes a single constraint, and none held by multipliers.";

/// What an index of a prescribed constraint is held at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Prescription {
    /// Its initial guess value.
    Initial,
    /// Its initial guess value plus an amount.
    Offset(Scalar),
    /// A value.
    Absolute(Scalar),
}

/// The rows that multipliers hold, beside what is eliminated.
pub(crate) enum Rows {
    /// No rows.
    None,
    /// Rows of a dense matrix.
    Dense(Matrix, Vector),
    /// Rows of a sparse matrix.
    Sparse(CscMatrix, Vector),
}

/// Equality constraints that all hold.
pub type EqualityConstraints = Vec<EqualityConstraint>;

/// Possible equality constraints.
#[derive(Clone)]
pub enum EqualityConstraint {
    /// Indices fixed at initial guess values.
    Fixed(Vec<usize>),
    /// Indices held at what they are prescribed.
    Prescribed(Vec<(usize, Prescription)>),
    /// Indices held at what they are prescribed by multipliers.
    Multipliers(Vec<(usize, Prescription)>),
    /// Linear equality constraint.
    Linear(Matrix, Vector),
    /// No constraint.
    None,
}

impl EqualityConstraint {
    /// The indices that are eliminated, which are those of a single fixed or prescribed
    /// constraint, once the constraints are merged.
    ///
    /// The indices of a prescribed constraint are in ascending order and without repeats.
    pub fn held_indices(constraints: &[Self]) -> Option<Vec<usize>> {
        let merged;
        let parts = if constraints.len() > 1 {
            merged = Self::merged(constraints.to_vec());
            merged.as_slice()
        } else {
            constraints
        };
        match parts {
            [Self::Fixed(indices)] => Some(indices.clone()),
            [Self::Prescribed(entries)] => Some(
                entries
                    .iter()
                    .map(|&(index, _)| index)
                    .collect::<BTreeSet<usize>>()
                    .into_iter()
                    .collect(),
            ),
            _ => None,
        }
    }
    /// Combines constraints into at most one prescribed, one with multipliers and one linear.
    ///
    /// The last entry for an index wins, whichever way it is held, and the linear constraints
    /// are stacked. A single constraint is left as it is.
    pub fn merged(constraints: Vec<Self>) -> Vec<Self> {
        if constraints.len() <= 1 {
            return constraints;
        }
        let mut eliminated = BTreeMap::new();
        let mut multiplied = BTreeMap::new();
        let mut rows = Vec::<(Matrix, Vector)>::new();
        constraints
            .into_iter()
            .for_each(|constraint| match constraint {
                Self::Fixed(indices) => indices.into_iter().for_each(|index| {
                    multiplied.remove(&index);
                    eliminated.insert(index, Prescription::Initial);
                }),
                Self::Prescribed(prescribed) => prescribed.into_iter().for_each(|(index, p)| {
                    multiplied.remove(&index);
                    eliminated.insert(index, p);
                }),
                Self::Multipliers(prescribed) => prescribed.into_iter().for_each(|(index, p)| {
                    eliminated.remove(&index);
                    multiplied.insert(index, p);
                }),
                Self::Linear(matrix, vector) => rows.push((matrix, vector)),
                Self::None => {}
            });
        let mut parts = vec![];
        if !eliminated.is_empty() {
            parts.push(Self::Prescribed(eliminated.into_iter().collect()));
        }
        if !multiplied.is_empty() {
            parts.push(Self::Multipliers(multiplied.into_iter().collect()));
        }
        if !rows.is_empty() {
            let width = rows[0].0.width();
            let height = rows.iter().map(|(matrix, _)| matrix.len()).sum();
            let mut matrix = Matrix::zero(height, width);
            let mut vector = Vector::zero(height);
            let mut offset = 0;
            rows.iter().for_each(|(block, rhs)| {
                assert_eq!(block.width(), width);
                (0..block.len()).for_each(|i| {
                    (0..width).for_each(|j| matrix[offset + i][j] = block[i][j]);
                    vector[offset + i] = rhs[i];
                });
                offset += block.len();
            });
            parts.push(Self::Linear(matrix, vector));
        }
        parts
    }
    /// What is prescribed relative to the initial guess, made relative to a reference instead.
    ///
    /// A solver that moves its initial guess as it steps through time uses this, so that what is
    /// held at a distance from where it starts is held at that distance from the reference.
    pub fn from_reference<X>(constraints: Vec<Self>, reference: &X) -> Vec<Self>
    where
        X: Solution,
    {
        let anchored = |entries: Vec<(usize, Prescription)>| {
            entries
                .into_iter()
                .map(|(index, prescription)| match prescription {
                    Prescription::Offset(offset) => (
                        index,
                        Prescription::Absolute(reference.flat_entry(index) + offset),
                    ),
                    prescription => (index, prescription),
                })
                .collect()
        };
        constraints
            .into_iter()
            .map(|constraint| match constraint {
                Self::Prescribed(entries) => Self::Prescribed(anchored(entries)),
                Self::Multipliers(entries) => Self::Multipliers(anchored(entries)),
                constraint => constraint,
            })
            .collect()
    }
    /// Eliminates what is prescribed, by moving the initial guess to it and fixing the indices.
    ///
    /// The last entry for an index wins, and the indices are fixed in ascending order.
    /// Other constraints are left as they are, after [`Self::merged`].
    pub fn lower<X>(constraints: Vec<Self>, initial_guess: &mut X) -> Vec<Self>
    where
        X: Solution,
    {
        Self::merged(constraints)
            .into_iter()
            .map(|constraint| match constraint {
                Self::Prescribed(entries) => {
                    let entries: BTreeMap<usize, Prescription> = entries.into_iter().collect();
                    entries
                        .iter()
                        .for_each(|(&index, prescription)| match prescription {
                            Prescription::Initial => {}
                            Prescription::Offset(offset) => initial_guess
                                .set_flat_entry(index, initial_guess.flat_entry(index) + offset),
                            Prescription::Absolute(value) => {
                                initial_guess.set_flat_entry(index, *value)
                            }
                        });
                    Self::Fixed(entries.into_keys().collect())
                }
                constraint => constraint,
            })
            .collect()
    }
    /// The one constraint the constraints are, once lowered, or nothing if they are several.
    pub(crate) fn lower_single<X>(constraints: Vec<Self>, initial_guess: &mut X) -> Option<Self>
    where
        X: Solution,
    {
        let mut parts = Self::lower(constraints, initial_guess);
        match parts.len() {
            0 => Some(Self::None),
            1 => parts.pop(),
            _ => None,
        }
    }
    /// The indices that are eliminated, with the guess moved to them, and the rows that
    /// multipliers hold, those of the multiplied indices first and then the linear ones.
    ///
    /// A multiplied index is held at its value relative to the guess as it was given.
    pub(crate) fn split<X>(
        constraints: Vec<Self>,
        initial_guess: &mut X,
    ) -> (Option<Vec<usize>>, Rows)
    where
        X: Solution,
    {
        let mut eliminated = None;
        let mut multiplied = vec![];
        let mut linear = None;
        Self::lower(constraints, initial_guess)
            .into_iter()
            .for_each(|part| match part {
                Self::Fixed(indices) => eliminated = Some(indices),
                Self::Multipliers(entries) => multiplied = entries,
                Self::Linear(matrix, vector) => linear = Some((matrix, vector)),
                Self::None => {}
                Self::Prescribed(_) => {
                    unreachable!("Prescribed constraints are lowered and merged.")
                }
            });
        if multiplied.is_empty() {
            return match linear {
                Some((matrix, vector)) => (eliminated, Rows::Dense(matrix, vector)),
                None => (eliminated, Rows::None),
            };
        }
        let width = initial_guess.size();
        let held = multiplied.len();
        let (matrix, rhs) = linear.unwrap_or_else(|| (Matrix::zero(0, width), Vector::zero(0)));
        assert!(matrix.is_empty() || matrix.width() == width);
        let mut pattern: Vec<(usize, usize)> = multiplied
            .iter()
            .enumerate()
            .map(|(row, &(index, _))| (row, index))
            .collect();
        (0..matrix.len()).for_each(|row| {
            (0..matrix.width()).for_each(|column| {
                if matrix[row][column] != 0.0 {
                    pattern.push((held + row, column))
                }
            })
        });
        let mut rows = CscMatrix::from_pattern(held + matrix.len(), width, pattern);
        rows.fill(|row, column| {
            if row < held {
                1.0
            } else {
                matrix[row - held][column]
            }
        });
        let mut values = Vector::zero(held + matrix.len());
        multiplied
            .iter()
            .enumerate()
            .for_each(|(row, &(index, prescription))| {
                let guess = initial_guess.flat_entry(index);
                values[row] = match prescription {
                    Prescription::Initial => guess,
                    Prescription::Offset(offset) => guess + offset,
                    Prescription::Absolute(value) => value,
                }
            });
        (0..matrix.len()).for_each(|row| values[held + row] = rhs[row]);
        (eliminated, Rows::Sparse(rows, values))
    }
}
