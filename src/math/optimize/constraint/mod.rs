#[cfg(test)]
mod test;

use crate::math::{Matrix, Scalar, Solution, Vector};
use std::collections::{BTreeMap, BTreeSet};

/// Why a prescribed and a linear constraint together are not solved yet.
pub(super) const ALL_UNSUPPORTED: &str =
    "Fixed and linear constraints together are not supported yet.";

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

/// Possible equality constraints.
#[derive(Clone)]
pub enum EqualityConstraint {
    /// Indices fixed at initial guess values.
    Fixed(Vec<usize>),
    /// Indices held at what they are prescribed.
    Prescribed(Vec<(usize, Prescription)>),
    /// Linear equality constraint.
    Linear(Matrix, Vector),
    /// Every one of the constraints.
    All(Vec<EqualityConstraint>),
    /// No constraint.
    None,
}

impl EqualityConstraint {
    /// The indices that are eliminated, which are those of a fixed or prescribed constraint.
    ///
    /// The indices of a prescribed constraint are in ascending order and without repeats.
    pub fn held_indices(&self) -> Option<Vec<usize>> {
        match self {
            Self::Fixed(indices) => Some(indices.clone()),
            Self::Prescribed(entries) => Some(
                entries
                    .iter()
                    .map(|&(index, _)| index)
                    .collect::<BTreeSet<usize>>()
                    .into_iter()
                    .collect(),
            ),
            Self::All(_) => match self.clone().merged() {
                Self::All(_) | Self::Linear(..) => None,
                merged => merged.held_indices(),
            },
            Self::Linear(..) | Self::None => None,
        }
    }
    /// Combines the constraints of [`Self::All`] into at most one prescribed and one linear.
    ///
    /// The last entry for an index wins, and the linear constraints are stacked.
    /// Other constraints are left as they are.
    pub fn merged(self) -> Self {
        let Self::All(constraints) = self else {
            return self;
        };
        let mut entries = BTreeMap::new();
        let mut rows = Vec::<(Matrix, Vector)>::new();
        let mut stack = constraints;
        stack.reverse();
        while let Some(constraint) = stack.pop() {
            match constraint {
                Self::Fixed(indices) => indices.into_iter().for_each(|index| {
                    entries.insert(index, Prescription::Initial);
                }),
                Self::Prescribed(prescribed) => prescribed.into_iter().for_each(|(index, p)| {
                    entries.insert(index, p);
                }),
                Self::Linear(matrix, vector) => rows.push((matrix, vector)),
                Self::All(inner) => stack.extend(inner.into_iter().rev()),
                Self::None => {}
            }
        }
        let prescribed =
            (!entries.is_empty()).then(|| Self::Prescribed(entries.into_iter().collect()));
        let linear = (!rows.is_empty()).then(|| {
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
            Self::Linear(matrix, vector)
        });
        match (prescribed, linear) {
            (Some(prescribed), Some(linear)) => Self::All(vec![prescribed, linear]),
            (Some(constraint), None) | (None, Some(constraint)) => constraint,
            (None, None) => Self::None,
        }
    }
    /// Eliminates what is prescribed, by moving the initial guess to it and fixing the indices.
    ///
    /// The last entry for an index wins, and the indices are fixed in ascending order.
    /// Other constraints are left as they are, after [`Self::merged`].
    pub fn lower<X>(self, initial_guess: &mut X) -> Self
    where
        X: Solution,
    {
        match self.merged() {
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
            Self::All(constraints) => Self::All(
                constraints
                    .into_iter()
                    .map(|constraint| constraint.lower(initial_guess))
                    .collect(),
            ),
            constraint => constraint,
        }
    }
}
