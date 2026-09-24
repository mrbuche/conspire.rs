use crate::{
    domain::{
        ElementModel, ElementModelError, Model, NodalCoordinates, NodalCoordinatesHistory,
        block::{element::Elements, finalize_node_neighbors, solver_from_neighbors},
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::{
        Jacobian, Tensor, TensorVec, Vector,
        optimize::{
            EqualityConstraint, FirstOrderRootFinding, FirstOrderRootFindingBlock, NewtonRaphson,
            OptimizationError, SolveStrategy,
        },
        sparse::{CscMatrix, SparseSolver},
    },
};
use std::cell::RefCell;

/// The assembled residuals and tangent blocks of the monolithic system. The unknowns
/// are the nodal coordinates, then the local unknowns of every integration point of
/// every element in turn.
pub struct MonolithicSystem {
    pub residual_global: Vector,
    pub residual_local: Vector,
    pub tangent_uu: CscMatrix,
    pub tangent_uv: CscMatrix,
    pub tangent_vu: CscMatrix,
    pub tangent_vv: CscMatrix,
}

impl MonolithicSystem {
    pub fn num_global(&self) -> usize {
        self.tangent_uu.height()
    }
    pub fn num_local(&self) -> usize {
        self.tangent_vv.height()
    }
    pub(crate) fn clear(&mut self) {
        self.residual_global = Vector::zero(self.num_global());
        self.residual_local = Vector::zero(self.num_local());
        self.tangent_uu.clear();
        self.tangent_uv.clear();
        self.tangent_vu.clear();
        self.tangent_vv.clear();
    }
    /// The positions of the tangent of the monolithic system, unknowns ordered as the
    /// nodal coordinates, then `constraints` multipliers, then the local unknowns.
    pub fn pattern(&self, constraints: usize) -> Vec<(usize, usize)> {
        let num_outer = self.num_global() + constraints;
        let mut pattern = self.tangent_uu.pattern().to_vec();
        pattern.extend(
            self.tangent_uv
                .pattern()
                .iter()
                .map(|&(row, column)| (row, num_outer + column)),
        );
        pattern.extend(
            self.tangent_vu
                .pattern()
                .iter()
                .map(|&(row, column)| (num_outer + row, column)),
        );
        pattern.extend(
            self.tangent_vv
                .pattern()
                .iter()
                .map(|&(row, column)| (num_outer + row, num_outer + column)),
        );
        pattern
    }
}

const MONOLITHIC_UNSUPPORTED: &str = "The monolithic solve is not supported for this domain.";

fn monolithic_unsupported() -> ElementModelError {
    ElementModelError::Upstream(MONOLITHIC_UNSUPPORTED.to_string(), String::new())
}

/// Assembly for rate-independent elastic-plastic solids, with the local unknowns of the
/// plastic step at each integration point converged by the local solver and eliminated
/// there, so the force and the stiffness of a point come from the same solve.
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
        local_solver: &NewtonRaphson,
        nodal_forces: &mut NodalForcesSolid<D>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError>;
    fn nodal_forces_and_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        local_solver: &NewtonRaphson,
    ) -> Result<(NodalForcesSolid<D>, NodalStiffnessesSolid<D>), ElementModelError> {
        let mut nodal_forces = NodalForcesSolid::zero(nodal_coordinates.len());
        let mut nodal_stiffnesses = NodalStiffnessesSolid::zero(nodal_coordinates.len());
        self.nodal_forces_and_stiffnesses_into(
            nodal_coordinates,
            state_variables,
            local_solver,
            &mut nodal_forces,
            &mut nodal_stiffnesses,
        )?;
        Ok((nodal_forces, nodal_stiffnesses))
    }
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        local_solver: &NewtonRaphson,
    ) -> Result<NodalForcesSolid<D>, ElementModelError> {
        Ok(self
            .nodal_forces_and_stiffnesses(nodal_coordinates, state_variables, local_solver)?
            .0)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        local_solver: &NewtonRaphson,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError> {
        Ok(self
            .nodal_forces_and_stiffnesses(nodal_coordinates, state_variables, local_solver)?
            .1)
    }
    /// An empty monolithic system holding the sparsity structure of its blocks, or
    /// `None` for a domain that does not support [`SolveStrategy::Monolithic`].
    fn monolithic_system(&self, _num_nodes: usize) -> Option<MonolithicSystem> {
        None
    }
    /// Evaluates the monolithic system at the given coordinates and trial local unknowns.
    fn monolithic_into(
        &self,
        _nodal_coordinates: &NodalCoordinates<D>,
        _state_variables: &S,
        _local: &Vector,
        _system: &mut MonolithicSystem,
    ) -> Result<(), ElementModelError> {
        Err(monolithic_unsupported())
    }
    /// The plastic state the local unknowns of a monolithic solve arrive at.
    fn monolithic_state(
        &self,
        _state_variables: &S,
        _local: &Vector,
    ) -> Result<S, ElementModelError> {
        Err(monolithic_unsupported())
    }
    /// Commit the plastic state at the converged coordinates of a load step.
    fn updated_state(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        local_solver: &NewtonRaphson,
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
        local_solver: &NewtonRaphson,
        nodal_forces: &mut NodalForcesSolid<D>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.blocks().nodal_forces_and_stiffnesses_into(
            nodal_coordinates,
            state_variables,
            local_solver,
            nodal_forces,
            nodal_stiffnesses,
        )
    }
    fn monolithic_system(&self, num_nodes: usize) -> Option<MonolithicSystem> {
        self.blocks().monolithic_system(num_nodes)
    }
    fn monolithic_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        local: &Vector,
        system: &mut MonolithicSystem,
    ) -> Result<(), ElementModelError> {
        self.blocks()
            .monolithic_into(nodal_coordinates, state_variables, local, system)
    }
    fn monolithic_state(
        &self,
        state_variables: &S,
        local: &Vector,
    ) -> Result<S, ElementModelError> {
        self.blocks().monolithic_state(state_variables, local)
    }
    fn updated_state(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        local_solver: &NewtonRaphson,
    ) -> Result<S, ElementModelError> {
        self.blocks()
            .updated_state(nodal_coordinates, state_variables, local_solver)
    }
}

/// Static load-stepping: solve nodal equilibrium with the plastic state of the previous
/// step, then commit the updated state, once per prescribed boundary condition.
///
/// With [`SolveStrategy::Condensed`] the local unknowns of the plastic step are converged
/// and eliminated at each integration point, so the solver only sees the nodal
/// coordinates. With [`SolveStrategy::Monolithic`] they are unknowns of the solve too,
/// stepped together with the nodal coordinates through one sparse system, with or
/// without eliminating them; that needs a domain that supports it, and only
/// [`EqualityConstraint::Linear`] boundary conditions.
pub trait ElasticPlasticRoot<S, const D: usize> {
    fn root(
        &self,
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        > + FirstOrderRootFindingBlock<
            Vector,
            Vector,
            Vector,
            Vector,
            CscMatrix,
            CscMatrix,
            CscMatrix,
            CscMatrix,
        >,
        boundary_conditions: &[EqualityConstraint],
        strategy: SolveStrategy,
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
        > + FirstOrderRootFindingBlock<
            Vector,
            Vector,
            Vector,
            Vector,
            CscMatrix,
            CscMatrix,
            CscMatrix,
            CscMatrix,
        >,
        boundary_conditions: &[EqualityConstraint],
        strategy: SolveStrategy,
    ) -> Result<(NodalCoordinatesHistory<D>, Vec<S>), OptimizationError> {
        let mut nodal_coordinates: NodalCoordinates<D> = self.coordinates().clone().into();
        let mut state = self.blocks().initial_state();
        let mut coordinates_history = NodalCoordinatesHistory::new();
        let mut state_history = Vec::new();
        coordinates_history.push(nodal_coordinates.clone());
        state_history.push(state.clone());
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let mut system = match strategy {
            SolveStrategy::Condensed(_) => None,
            SolveStrategy::Monolithic { .. } => Some(
                self.blocks()
                    .monolithic_system(nodal_coordinates.len())
                    .ok_or_else(|| MONOLITHIC_UNSUPPORTED.to_string())?,
            ),
        };
        for constraint in boundary_conditions {
            let frozen_state = state.clone();
            if let (Some(monolithic), SolveStrategy::Monolithic { elimination }) =
                (system.take(), &strategy)
            {
                let EqualityConstraint::Linear(matrix, vector) = constraint else {
                    return Err(OptimizationError::Intermediate(
                        "The monolithic solve only supports EqualityConstraint::Linear."
                            .to_string(),
                    ));
                };
                let (num_global, num_local) = (monolithic.num_global(), monolithic.num_local());
                //
                // Eliminating the local unknowns leaves the sparse solver the global
                // system alone, whose pattern is that of the stiffness.
                //
                let mut pattern = if *elimination {
                    monolithic.tangent_uu.pattern().to_vec()
                } else {
                    monolithic.pattern(matrix.len())
                };
                let mut constraint_pattern = Vec::new();
                (0..matrix.len()).for_each(|row| {
                    (0..matrix.width()).for_each(|column| {
                        if matrix[row][column] != 0.0 {
                            constraint_pattern.push((row, column));
                            pattern.push((num_global + row, column));
                            pattern.push((column, num_global + row))
                        }
                    })
                });
                pattern.sort_unstable();
                pattern.dedup();
                let sparse = SparseSolver::from_pattern(
                    num_global + matrix.len() + if *elimination { 0 } else { num_local },
                    pattern,
                    false,
                );
                let mut constraint_matrix =
                    CscMatrix::from_pattern(matrix.len(), matrix.width(), constraint_pattern);
                constraint_matrix.fill(|row, column| matrix[row][column]);
                let mut initial = Vector::zero(num_global);
                nodal_coordinates.fill_into(&mut initial);
                //
                // The three closures are called at the same point in a row, so one
                // evaluation of the system serves them all.
                //
                let cache: RefCell<Option<(Vector, Vector)>> = RefCell::new(None);
                let evaluate = |global: &Vector,
                                local: &Vector,
                                system: &mut MonolithicSystem|
                 -> Result<(), String> {
                    let mut cache = cache.borrow_mut();
                    if let Some((cached_global, cached_local)) = cache.as_ref()
                        && cached_global == global
                        && cached_local == local
                    {
                        return Ok(());
                    }
                    self.blocks()
                        .monolithic_into(
                            &NodalCoordinates::from(global.clone()),
                            &frozen_state,
                            local,
                            system,
                        )
                        .map_err(|error| error.to_string())?;
                    *cache = Some((global.clone(), local.clone()));
                    Ok(())
                };
                let cell = RefCell::new(monolithic);
                let (new_global, new_local) = solver.root_block(
                    |global: &Vector, local: &Vector| {
                        let mut system = cell.borrow_mut();
                        evaluate(global, local, &mut system)?;
                        Ok(system.residual_global.clone())
                    },
                    |global: &Vector, local: &Vector| {
                        let mut system = cell.borrow_mut();
                        evaluate(global, local, &mut system)?;
                        Ok(system.residual_local.clone())
                    },
                    |global: &Vector, local: &Vector| {
                        let mut system = cell.borrow_mut();
                        evaluate(global, local, &mut system)?;
                        Ok((
                            system.tangent_uu.clone(),
                            system.tangent_vu.clone(),
                            system.tangent_uv.clone(),
                            system.tangent_vv.clone(),
                        ))
                    },
                    (initial, Vector::zero(num_local)),
                    (constraint_matrix, vector.clone()),
                    (
                        CscMatrix::from_pattern(0, num_local, Vec::new()),
                        Vector::zero(0),
                    ),
                    Some(sparse),
                    strategy.clone(),
                )?;
                nodal_coordinates = NodalCoordinates::from(new_global);
                state = self
                    .blocks()
                    .monolithic_state(&frozen_state, &new_local)
                    .map_err(|error| error.to_string())?;
                system = Some(cell.into_inner());
            } else if let SolveStrategy::Condensed(local_solver) = &strategy {
                let sparse = solver_from_neighbors(&neighbors, constraint, D, false);
                //
                // The force and the stiffness are asked for at the same coordinates in
                // a row, and both come from one solve of every integration point.
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
                        .nodal_forces_and_stiffnesses(coordinates, &frozen_state, local_solver)
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
                    .updated_state(&nodal_coordinates, &frozen_state, local_solver)
                    .map_err(|error| error.to_string())?;
            }
            coordinates_history.push(nodal_coordinates.clone());
            state_history.push(state.clone());
        }
        Ok((coordinates_history, state_history))
    }
}
