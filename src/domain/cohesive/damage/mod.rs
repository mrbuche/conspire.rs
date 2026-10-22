use crate::{
    domain::{
        ElementModelError, Model, NodalCoordinates,
        block::element::Elements,
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::{Tensor, integrate::StepError},
};

/// Assembly for cohesive interfaces that keep the greatest opening reached at each of
/// their integration points, which only grows.
pub trait DamageElements<S, const D: usize>
where
    Self: Elements,
{
    /// The undamaged state.
    fn initial_state(&self) -> S;
    /// Adds the nodal forces at the given state.
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state: &S,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError>;
    /// Computes the nodal forces at the given state.
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state: &S,
    ) -> Result<NodalForcesSolid<D>, ElementModelError> {
        let mut nodal_forces = NodalForcesSolid::zero(nodal_coordinates.len());
        self.nodal_forces_into(nodal_coordinates, state, &mut nodal_forces)?;
        Ok(nodal_forces)
    }
    /// Adds the nodal stiffnesses at the given state.
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state: &S,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError>;
    /// Computes the nodal stiffnesses at the given state.
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state: &S,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError> {
        let mut nodal_stiffnesses = NodalStiffnessesSolid::zero(nodal_coordinates.len());
        self.nodal_stiffnesses_into(nodal_coordinates, state, &mut nodal_stiffnesses)?;
        Ok(nodal_stiffnesses)
    }
    /// Commits the state at the converged coordinates of a load step.
    fn updated_state(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state: &S,
    ) -> Result<S, ElementModelError>;
    /// The error indicators of a load step, from the coordinates and the states at its
    /// two ends, the larger over the integration points.
    fn step_error(
        &self,
        nodal_coordinates_old: &NodalCoordinates<D>,
        nodal_coordinates: &NodalCoordinates<D>,
        state: &S,
        updated_state: &S,
    ) -> Result<StepError, ElementModelError>;
}

impl<B, S, const D: usize> DamageElements<S, D> for Model<B, D>
where
    B: DamageElements<S, D>,
{
    fn initial_state(&self) -> S {
        self.blocks().initial_state()
    }
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state: &S,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.blocks()
            .nodal_forces_into(nodal_coordinates, state, nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state: &S,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.blocks()
            .nodal_stiffnesses_into(nodal_coordinates, state, nodal_stiffnesses)
    }
    fn updated_state(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state: &S,
    ) -> Result<S, ElementModelError> {
        self.blocks().updated_state(nodal_coordinates, state)
    }
    fn step_error(
        &self,
        nodal_coordinates_old: &NodalCoordinates<D>,
        nodal_coordinates: &NodalCoordinates<D>,
        state: &S,
        updated_state: &S,
    ) -> Result<StepError, ElementModelError> {
        self.blocks().step_error(
            nodal_coordinates_old,
            nodal_coordinates,
            state,
            updated_state,
        )
    }
}
