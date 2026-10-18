use super::NodalTemperaturesHistory;
use crate::{
    domain::{
        boundary::{ConstrainAt, Field},
        thermal::{NodalTemperatures, capacity::HeatCapacityMatrix},
    },
    math::{
        Quantity, Tensor,
        integrate::{IntegrationError, Times},
        optimize::{EqualityConstraint, EqualityConstraints, Prescription},
    },
    units::{Temperature, Time},
};
use std::collections::{BTreeMap, BTreeSet};

/// The temperatures of a model held in place, and the motion of those that are moved.
///
/// The conditions are on the temperatures of the nodes, indexed by node. What is held at a
/// distance from where it starts is held at that distance from the temperatures the motion
/// starts at, whatever the step it is held in. The rates of the temperatures that are moved
/// are not given by an explicit scheme, which does not integrate them.
pub(super) struct Motion<'c, C> {
    conditions: &'c C,
    anchor: NodalTemperatures,
    moving: Vec<usize>,
}

impl<'c, C> Motion<'c, C>
where
    C: ConstrainAt<Field<Temperature>>,
{
    /// The inverse of the heat capacities with the held temperatures removed, and the motion,
    /// once the temperatures are moved to where the motion starts.
    pub(super) fn held<M>(
        capacities: &M,
        conditions: &'c C,
        initial_time: Quantity<Time>,
        temperatures: &mut NodalTemperatures,
        integrator: &str,
    ) -> Result<(M::Inverse, Self), IntegrationError>
    where
        M: HeatCapacityMatrix,
    {
        let mut fixed = BTreeSet::new();
        let mut moving = BTreeSet::new();
        for constraint in conditions.constraint_at(initial_time) {
            match constraint {
                EqualityConstraint::None => {}
                EqualityConstraint::Fixed(indices) => fixed.extend(indices),
                EqualityConstraint::Prescribed(entries) => {
                    entries.into_iter().for_each(|(index, prescription)| {
                        if prescription == Prescription::Initial {
                            fixed.insert(index);
                        } else {
                            moving.insert(index);
                        }
                    })
                }
                EqualityConstraint::Multipliers(_) => {
                    return Err(IntegrationError::Intermediate(format!(
                        "Constraints held by multipliers are not supported by {integrator}."
                    )));
                }
                EqualityConstraint::Linear(..) => {
                    return Err(IntegrationError::Intermediate(format!(
                        "Linear constraints are not supported by {integrator}."
                    )));
                }
            }
        }
        moving.retain(|node| !fixed.contains(node));
        let held: Vec<usize> = fixed.union(&moving).copied().collect();
        let inverse = capacities
            .inverse(&held)
            .map_err(|error| IntegrationError::Intermediate(error.to_string()))?;
        let motion = Self {
            conditions,
            anchor: temperatures.clone(),
            moving: moving.into_iter().collect(),
        };
        motion.apply(initial_time, temperatures);
        Ok((inverse, motion))
    }
    /// The constraints at a time, with what is held at a distance from where it starts held at
    /// that distance from the temperatures the motion starts at.
    pub(super) fn constraint_at(&self, time: Quantity<Time>) -> EqualityConstraints {
        EqualityConstraint::from_reference(self.conditions.constraint_at(time), &self.anchor)
    }
    /// What the moved temperatures are held at, at a time.
    fn targets(&self, time: Quantity<Time>) -> BTreeMap<usize, f64> {
        let mut targets = BTreeMap::new();
        for constraint in self.constraint_at(time) {
            if let EqualityConstraint::Prescribed(entries) = constraint {
                for (node, prescription) in entries {
                    if let Prescription::Absolute(value) = prescription {
                        targets.insert(node, value);
                    }
                }
            }
        }
        targets
    }
    /// Moves the temperatures that are moved to where they are held at a time.
    pub(super) fn apply(&self, time: Quantity<Time>, temperatures: &mut NodalTemperatures) {
        let targets = self.targets(time);
        self.moving.iter().for_each(|node| {
            let start = self.anchor[*node].value();
            temperatures[*node] = Quantity::new(targets.get(node).copied().unwrap_or(start));
        });
    }
    /// Gives the temperatures that are moved the motion they are held at in the history.
    pub(super) fn record(&self, times: &Times, temperatures: &mut NodalTemperaturesHistory) {
        times
            .iter()
            .take(temperatures.len())
            .enumerate()
            .for_each(|(k, &time)| {
                self.apply(time, &mut temperatures[k]);
            });
    }
}
