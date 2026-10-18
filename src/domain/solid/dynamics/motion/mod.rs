use crate::{
    domain::{
        NodalCoordinates, NodalCoordinatesHistory, NodalVelocities,
        boundary::{ConstrainAt, Solid},
        solid::mass::MassMatrix,
    },
    math::{
        Quantity, Tensor,
        integrate::{IntegrationError, Times},
        optimize::{EqualityConstraint, EqualityConstraints, Prescription},
    },
    units::Time,
};
use std::collections::{BTreeMap, BTreeSet};

/// The degrees of freedom of a solid held in place, and the motion of those that are moved.
///
/// The conditions are on the coordinates of the nodes, indexed as `3 * node + component`. What
/// is held at a distance from where it starts is held at that distance from the coordinates
/// the motion starts at, whatever the step it is held in. Fixed degrees of freedom have no
/// velocity, and those that are moved keep the one they are given, which a scheme that
/// moves them itself uses, and an explicit one does not report.
pub(super) struct Motion<'c, C> {
    conditions: &'c C,
    anchor: NodalCoordinates<3>,
    moving: Vec<usize>,
}

impl<'c, C> Motion<'c, C>
where
    C: ConstrainAt<Solid<3>>,
{
    /// The masses with the held degrees of freedom removed, the velocities with them removed,
    /// and the motion, once the coordinates are moved to where the motion starts.
    pub(super) fn held<M>(
        masses: &M,
        conditions: &'c C,
        initial_time: Quantity<Time>,
        coordinates: &mut NodalCoordinates<3>,
        mut velocities: NodalVelocities<3>,
    ) -> Result<(M::Inverse, NodalVelocities<3>, Self), IntegrationError>
    where
        M: MassMatrix<3>,
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
                    return Err(IntegrationError::Intermediate(
                        "Constraints held by multipliers are not supported by dynamics."
                            .to_string(),
                    ));
                }
                EqualityConstraint::Linear(..) => {
                    return Err(IntegrationError::Intermediate(
                        "Linear constraints are not supported by dynamics.".to_string(),
                    ));
                }
            }
        }
        moving.retain(|dof| !fixed.contains(dof));
        let held: Vec<usize> = fixed.union(&moving).copied().collect();
        let inverse = masses
            .inverse(&held)
            .map_err(|error| IntegrationError::Intermediate(error.to_string()))?;
        fixed
            .iter()
            .for_each(|&dof| velocities[dof / 3][dof % 3] = Default::default());
        let motion = Self {
            conditions,
            anchor: coordinates.clone(),
            moving: moving.into_iter().collect(),
        };
        motion.apply(initial_time, coordinates);
        Ok((inverse, velocities, motion))
    }
    /// The constraints at a time, with what is held at a distance from where it starts held at
    /// that distance from the coordinates the motion starts at.
    pub(super) fn constraint_at(&self, time: Quantity<Time>) -> EqualityConstraints {
        EqualityConstraint::from_reference(self.conditions.constraint_at(time), &self.anchor)
    }
    /// What the moved degrees of freedom are held at, at a time.
    fn targets(&self, time: Quantity<Time>) -> BTreeMap<usize, f64> {
        let mut targets = BTreeMap::new();
        for constraint in self.conditions.constraint_at(time) {
            if let EqualityConstraint::Prescribed(entries) = constraint {
                for (dof, prescription) in entries {
                    let start = self.anchor[dof / 3][dof % 3].value();
                    match prescription {
                        Prescription::Initial => {}
                        Prescription::Offset(offset) => {
                            targets.insert(dof, start + offset);
                        }
                        Prescription::Absolute(value) => {
                            targets.insert(dof, value);
                        }
                    }
                }
            }
        }
        targets
    }
    /// Moves the degrees of freedom that are moved to where they are held at a time.
    pub(super) fn apply(&self, time: Quantity<Time>, coordinates: &mut NodalCoordinates<3>) {
        let targets = self.targets(time);
        self.moving.iter().for_each(|dof| {
            let start = self.anchor[dof / 3][dof % 3].value();
            coordinates[dof / 3][dof % 3] =
                Quantity::new(targets.get(dof).copied().unwrap_or(start));
        });
    }
    /// Gives the degrees of freedom that are moved the motion they are held at in the history.
    pub(super) fn record(&self, times: &Times, coordinates: &mut NodalCoordinatesHistory<3>) {
        times.iter().enumerate().for_each(|(k, &time)| {
            self.apply(time, &mut coordinates[k]);
        });
    }
}
