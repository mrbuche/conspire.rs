use crate::{
    domain::{
        NodalCoordinates, NodalCoordinatesHistory,
        boundary::{ConstrainAt, SolidThermal},
        multiphysics::solid_thermal::{NodalStatesSolidThermal, join_states},
        thermal::{NodalTemperatures, transient::NodalTemperaturesHistory},
    },
    math::{
        Quantity, Tensor,
        integrate::{IntegrationError, Times},
        optimize::{EqualityConstraint, EqualityConstraints, Prescription},
    },
    units::Time,
};
use std::collections::{BTreeMap, BTreeSet};

/// The coordinates and temperatures of a thermoelastic model held in place, and the motion of
/// those that are moved.
///
/// The conditions are on the coordinates and temperatures of the nodes, indexed as `4 * node +
/// component` with the temperature as the last component. What is held at a distance from
/// where it starts is held at that distance from the coordinates and temperatures the motion
/// starts at, whatever the step it is held in.
pub(super) struct Motion<'c, C> {
    conditions: &'c C,
    anchor: NodalStatesSolidThermal,
    fixed_coordinates: Vec<usize>,
    moving_coordinates: Vec<usize>,
    fixed_temperatures: Vec<usize>,
    moving_temperatures: Vec<usize>,
}

impl<'c, C> Motion<'c, C>
where
    C: ConstrainAt<SolidThermal<3>>,
{
    /// The motion, once the coordinates and temperatures are moved to where the motion starts.
    pub(super) fn new(
        conditions: &'c C,
        initial_time: Quantity<Time>,
        coordinates: &mut NodalCoordinates<3>,
        temperatures: &mut NodalTemperatures,
        integrator: &str,
    ) -> Result<Self, IntegrationError> {
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
        moving.retain(|index| !fixed.contains(index));
        let split = |indices: BTreeSet<usize>| {
            let coordinates = indices
                .iter()
                .filter(|&&index| index % 4 < 3)
                .map(|&index| 3 * (index / 4) + index % 4)
                .collect();
            let temperatures = indices
                .iter()
                .filter(|&&index| index % 4 == 3)
                .map(|&index| index / 4)
                .collect();
            (coordinates, temperatures)
        };
        let (fixed_coordinates, fixed_temperatures) = split(fixed);
        let (moving_coordinates, moving_temperatures) = split(moving);
        let motion = Self {
            conditions,
            anchor: join_states(coordinates, temperatures),
            fixed_coordinates,
            moving_coordinates,
            fixed_temperatures,
            moving_temperatures,
        };
        motion.apply(initial_time, coordinates, temperatures);
        Ok(motion)
    }
    /// The degrees of freedom of the coordinates that are fixed, indexed as `3 * node +
    /// component`.
    pub(super) fn fixed_coordinates(&self) -> &[usize] {
        &self.fixed_coordinates
    }
    /// The degrees of freedom of the coordinates that are fixed or moved.
    pub(super) fn held_coordinates(&self) -> Vec<usize> {
        union(&self.fixed_coordinates, &self.moving_coordinates)
    }
    /// The nodes with temperatures that are fixed or moved.
    pub(super) fn held_temperatures(&self) -> Vec<usize> {
        union(&self.fixed_temperatures, &self.moving_temperatures)
    }
    /// The constraints at a time, with what is held at a distance from where it starts held at
    /// that distance from the state the motion starts at.
    pub(super) fn constraint_at(&self, time: Quantity<Time>) -> EqualityConstraints {
        EqualityConstraint::from_reference(self.conditions.constraint_at(time), &self.anchor)
    }
    /// What the moved degrees of freedom are held at, at a time, by their combined indices.
    fn targets(&self, time: Quantity<Time>) -> BTreeMap<usize, f64> {
        let mut targets = BTreeMap::new();
        for constraint in self.constraint_at(time) {
            if let EqualityConstraint::Prescribed(entries) = constraint {
                for (index, prescription) in entries {
                    if let Prescription::Absolute(value) = prescription {
                        targets.insert(index, value);
                    }
                }
            }
        }
        targets
    }
    /// Moves the coordinates and temperatures that are moved to where they are held at a time.
    pub(super) fn apply(
        &self,
        time: Quantity<Time>,
        coordinates: &mut NodalCoordinates<3>,
        temperatures: &mut NodalTemperatures,
    ) {
        let targets = self.targets(time);
        self.moving_coordinates.iter().for_each(|dof| {
            let combined = 4 * (dof / 3) + dof % 3;
            let (node, component) = (dof / 3, dof % 3);
            let start = coordinates[node][component].value();
            coordinates[node][component] =
                Quantity::new(targets.get(&combined).copied().unwrap_or(start));
        });
        self.moving_temperatures.iter().for_each(|&node| {
            let start = temperatures[node].value();
            temperatures[node] =
                Quantity::new(targets.get(&(4 * node + 3)).copied().unwrap_or(start));
        });
    }
    /// Gives the coordinates and temperatures that are moved the motion they are held at in the
    /// histories.
    pub(super) fn record(
        &self,
        times: &Times,
        coordinates: &mut NodalCoordinatesHistory<3>,
        temperatures: &mut NodalTemperaturesHistory,
    ) {
        let recorded = coordinates.len().min(temperatures.len());
        times
            .iter()
            .take(recorded)
            .enumerate()
            .for_each(|(k, &time)| {
                self.apply(time, &mut coordinates[k], &mut temperatures[k]);
            });
    }
}

fn union(a: &[usize], b: &[usize]) -> Vec<usize> {
    a.iter()
        .chain(b)
        .copied()
        .collect::<BTreeSet<usize>>()
        .into_iter()
        .collect()
}
