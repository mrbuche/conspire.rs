#[cfg(test)]
mod test;

use crate::math::{Matrix, Scalar, Solution, Vector};
use std::collections::{BTreeMap, BTreeSet};

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
            Self::Linear(..) | Self::None => None,
        }
    }
    /// Eliminates what is prescribed, by moving the initial guess to it and fixing the indices.
    ///
    /// The last entry for an index wins, and the indices are fixed in ascending order.
    /// Other constraints are left as they are.
    pub fn lower<X>(self, initial_guess: &mut X) -> Self
    where
        X: Solution,
    {
        match self {
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
        }
    }
}
