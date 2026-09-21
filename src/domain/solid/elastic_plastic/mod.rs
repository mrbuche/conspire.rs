use crate::{
    domain::{
        ElementModel, ElementModelError, Model, NodalCoordinates, NodalCoordinatesHistory,
        block::{element::Elements, finalize_node_neighbors, solver_from_neighbors},
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::{
        Tensor, TensorVec,
        optimize::{EqualityConstraint, FirstOrderRootFinding, OptimizationError},
    },
};
use std::cell::RefCell;

/// Assembly for rate-independent elastic-plastic solids, with the local unknowns of the
/// plastic step at each integration point converged and eliminated there, so the force
/// and the stiffness of a point come from the same solve.
pub trait ElasticPlasticElements<S, const D: usize>
where
    Self: Elements,
{
    /// The initial (unyielded) plastic state field.
    fn initial_state(&self) -> S;
    /// Adds the nodal forces and the nodal stiffnesses.
    fn nodal_forces_and_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        nodal_forces: &mut NodalForcesSolid<D>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError>;
    fn nodal_forces_and_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<(NodalForcesSolid<D>, NodalStiffnessesSolid<D>), ElementModelError> {
        let mut nodal_forces = NodalForcesSolid::zero(nodal_coordinates.len());
        let mut nodal_stiffnesses = NodalStiffnessesSolid::zero(nodal_coordinates.len());
        self.nodal_forces_and_stiffnesses_into(
            nodal_coordinates,
            state_variables,
            &mut nodal_forces,
            &mut nodal_stiffnesses,
        )?;
        Ok((nodal_forces, nodal_stiffnesses))
    }
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalForcesSolid<D>, ElementModelError> {
        Ok(self
            .nodal_forces_and_stiffnesses(nodal_coordinates, state_variables)?
            .0)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError> {
        Ok(self
            .nodal_forces_and_stiffnesses(nodal_coordinates, state_variables)?
            .1)
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
    fn nodal_forces_and_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        nodal_forces: &mut NodalForcesSolid<D>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.blocks().nodal_forces_and_stiffnesses_into(
            nodal_coordinates,
            state_variables,
            nodal_forces,
            nodal_stiffnesses,
        )
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

/// Static load-stepping: solve nodal equilibrium with the plastic state of the previous
/// step, then commit the updated state, once per prescribed boundary condition.
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

type Evaluation<const D: usize> = (
    NodalCoordinates<D>,
    NodalForcesSolid<D>,
    NodalStiffnessesSolid<D>,
);

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
            //
            // The force and the stiffness are asked for at the same coordinates in a
            // row, and both come from one solve of every integration point.
            //
            let cache: RefCell<Option<Evaluation<D>>> = RefCell::new(None);
            let evaluate = |coordinates: &NodalCoordinates<D>| -> Result<(), String> {
                let mut cache = cache.borrow_mut();
                if let Some((cached, ..)) = cache.as_ref()
                    && cached == coordinates
                {
                    return Ok(());
                }
                let (forces, stiffnesses) = self
                    .blocks()
                    .nodal_forces_and_stiffnesses(coordinates, &frozen_state)
                    .map_err(|error| error.to_string())?;
                *cache = Some((coordinates.clone(), forces, stiffnesses));
                Ok(())
            };
            nodal_coordinates = solver.root(
                |coordinates: &NodalCoordinates<D>| {
                    evaluate(coordinates)?;
                    Ok(cache.borrow().as_ref().unwrap().1.clone())
                },
                |coordinates: &NodalCoordinates<D>| {
                    evaluate(coordinates)?;
                    Ok(cache.borrow().as_ref().unwrap().2.clone())
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
