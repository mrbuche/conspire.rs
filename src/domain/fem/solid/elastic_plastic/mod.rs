use crate::{
    fem::{
        ElementModel, ElementModelError, Elements, Model, NodalCoordinates,
        NodalCoordinatesHistory,
        block::{finalize_node_neighbors, solver_from_neighbors},
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::{
        Tensor, TensorVec,
        optimize::{EqualityConstraint, FirstOrderRootFinding, OptimizationError},
    },
};

/// Assembly for rate-independent elastic-plastic solids, with the plastic state
/// condensed at each integration point.
pub trait ElasticPlasticElements<S, const D: usize>
where
    Self: Elements,
{
    /// The initial (unyielded) plastic state field.
    fn initial_state(&self) -> S;
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError>;
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalForcesSolid<D>, ElementModelError> {
        let mut nodal_forces = NodalForcesSolid::zero(nodal_coordinates.len());
        self.nodal_forces_into(nodal_coordinates, state_variables, &mut nodal_forces)?;
        Ok(nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError> {
        let mut nodal_stiffnesses = NodalStiffnessesSolid::zero(nodal_coordinates.len());
        self.nodal_stiffnesses_into(nodal_coordinates, state_variables, &mut nodal_stiffnesses)?;
        Ok(nodal_stiffnesses)
    }
    /// Commit the plastic state at the converged coordinates of a load step.
    fn updated_state(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<S, ElementModelError>;
}

impl<B, S, const D: usize> ElasticPlasticElements<S, D> for Model<B, D>
where
    B: ElasticPlasticElements<S, D>,
{
    fn initial_state(&self) -> S {
        self.blocks().initial_state()
    }
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.blocks()
            .nodal_forces_into(nodal_coordinates, state_variables, nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.blocks()
            .nodal_stiffnesses_into(nodal_coordinates, state_variables, nodal_stiffnesses)
    }
    fn updated_state(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<S, ElementModelError> {
        self.blocks()
            .updated_state(nodal_coordinates, state_variables)
    }
}

/// Static load-stepping: solve nodal equilibrium with the plastic state frozen, then
/// commit the return-mapped state, once per prescribed boundary condition.
pub trait ElasticPlasticRoot<S, const D: usize> {
    fn root(
        &self,
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        boundary_conditions: &[EqualityConstraint],
    ) -> Result<(NodalCoordinatesHistory<D>, Vec<S>), OptimizationError>;
}

impl<B, S, const D: usize> ElasticPlasticRoot<S, D> for Model<B, D>
where
    B: ElasticPlasticElements<S, D>,
    S: Clone,
{
    fn root(
        &self,
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        boundary_conditions: &[EqualityConstraint],
    ) -> Result<(NodalCoordinatesHistory<D>, Vec<S>), OptimizationError> {
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let mut nodal_coordinates: NodalCoordinates<D> = self.coordinates().clone().into();
        let mut state = self.blocks().initial_state();
        let mut coordinates_history = NodalCoordinatesHistory::new();
        let mut state_history = Vec::new();
        coordinates_history.push(nodal_coordinates.clone());
        state_history.push(state.clone());
        for constraint in boundary_conditions {
            let sparse = solver_from_neighbors(&neighbors, constraint, D, false);
            let frozen_state = state.clone();
            nodal_coordinates = solver.root(
                |coordinates: &NodalCoordinates<D>| {
                    self.blocks()
                        .nodal_forces(coordinates, &frozen_state)
                        .map_err(|error| error.to_string())
                },
                |coordinates: &NodalCoordinates<D>| {
                    self.blocks()
                        .nodal_stiffnesses(coordinates, &frozen_state)
                        .map_err(|error| error.to_string())
                },
                nodal_coordinates.clone(),
                constraint.clone(),
                Some(sparse),
            )?;
            state = self
                .blocks()
                .updated_state(&nodal_coordinates, &frozen_state)
                .map_err(|error| error.to_string())?;
            coordinates_history.push(nodal_coordinates.clone());
            state_history.push(state.clone());
        }
        Ok((coordinates_history, state_history))
    }
}
