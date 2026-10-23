use crate::{
    domain::{
        Blocks, ElementModel, ElementModelError, Model, NodalCoordinates, NodalCoordinatesHistory,
        NodalState,
        block::{element::Elements, finalize_node_neighbors, solver_from_neighbors},
        boundary::{ConstrainAt, LoadAt, LoadStiffnessAt},
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::{
        Jacobian, Quantity, Tensor, TensorVec, Vector,
        integrate::{StepError, adapt},
        optimize::{
            EqualityConstraint, NewtonRaphson, OptimizationError, Prescription, RootFinding,
            RootFindingBlock, SolveStrategy,
        },
        sparse::{CscMatrix, SparseSolver},
    },
    units::Time,
};
use std::{cell::RefCell, collections::BTreeMap};

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

/// Assembly for blocks with rate-independent history variables, such as the plastic state
/// of a solid or the greatest opening of a cohesive interface, which change only when a
/// load step is committed. The local unknowns of a plastic step at each integration point
/// are converged by the local solver and eliminated there, so the force and the stiffness
/// of a point come from the same solve; a block with none ignores the local solver.
///
/// A tuple of blocks has the tuple of their states.
pub trait RateIndependentElements<S, const D: usize>
where
    Self: Elements,
{
    /// The initial state field, such as unyielded or undamaged.
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
    /// Computes the nodal forces and the nodal stiffnesses.
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
    /// The error indicators of a load step, from the coordinates and the plastic states at
    /// its two ends, the larger over the integration points.
    fn step_error(
        &self,
        nodal_coordinates_old: &NodalCoordinates<D>,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        updated_state: &S,
    ) -> Result<StepError, ElementModelError>;
}

impl<B, S, const D: usize> RateIndependentElements<S, D> for Model<B, D>
where
    B: RateIndependentElements<S, D>,
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
    fn step_error(
        &self,
        nodal_coordinates_old: &NodalCoordinates<D>,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        updated_state: &S,
    ) -> Result<StepError, ElementModelError> {
        self.blocks().step_error(
            nodal_coordinates_old,
            nodal_coordinates,
            state_variables,
            updated_state,
        )
    }
}

impl<B1, B2, S1, S2, const D: usize> RateIndependentElements<(S1, S2), D> for Blocks<B1, B2>
where
    B1: RateIndependentElements<S1, D>,
    B2: RateIndependentElements<S2, D>,
{
    fn initial_state(&self) -> (S1, S2) {
        (self.0.initial_state(), self.1.initial_state())
    }
    fn nodal_forces_and_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        (state_1, state_2): &(S1, S2),
        local_solver: &NewtonRaphson,
        nodal_forces: &mut NodalForcesSolid<D>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.0.nodal_forces_and_stiffnesses_into(
            nodal_coordinates,
            state_1,
            local_solver,
            nodal_forces,
            nodal_stiffnesses,
        )?;
        self.1.nodal_forces_and_stiffnesses_into(
            nodal_coordinates,
            state_2,
            local_solver,
            nodal_forces,
            nodal_stiffnesses,
        )
    }
    fn updated_state(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        (state_1, state_2): &(S1, S2),
        local_solver: &NewtonRaphson,
    ) -> Result<(S1, S2), ElementModelError> {
        Ok((
            self.0
                .updated_state(nodal_coordinates, state_1, local_solver)?,
            self.1
                .updated_state(nodal_coordinates, state_2, local_solver)?,
        ))
    }
    fn step_error(
        &self,
        nodal_coordinates_old: &NodalCoordinates<D>,
        nodal_coordinates: &NodalCoordinates<D>,
        (state_1, state_2): &(S1, S2),
        (updated_state_1, updated_state_2): &(S1, S2),
    ) -> Result<StepError, ElementModelError> {
        Ok(self
            .0
            .step_error(
                nodal_coordinates_old,
                nodal_coordinates,
                state_1,
                updated_state_1,
            )?
            .max(self.1.step_error(
                nodal_coordinates_old,
                nodal_coordinates,
                state_2,
                updated_state_2,
            )?))
    }
}

/// Quasi-static adaptive load stepping: solve nodal equilibrium with the plastic state of
/// the last accepted step, then commit the updated state when the error of the step is
/// within tolerance.
///
/// The model is rate independent, so the times are values of the load parameter, and the
/// step is chosen by [`StepError`]. Two times give only the span, and the accepted steps
/// are reported. More than two are breakpoints: the steps are cut to end on each, and only
/// they are reported, since the response has kinks at yielding and at unloading that an
/// interpolant between steps would smooth over. The first time is where the initial
/// coordinates are. A step is accepted when both the estimated error of the step in the
/// plastic deformation, from the change of the flow direction over it, and the plastic
/// increment of a step that begins inside the yield surface, the larger of each over the
/// integration points, are within `abs_tol`, in units of the reversible range of the
/// response at the point, the yield strain of a plastic solid and the opening at which a
/// cohesive interface reaches its strength, so the tolerance is the same fraction of it for
/// every block, and a step the solver fails on is cut back like a rejected one. The tolerance bounds the estimated error of
/// each step, as those of the integrators do, so the errors add up over the steps: the
/// error of the result is larger than `abs_tol` and falls more slowly than it.
///
/// The conditions are held at each time, and what is held at a distance from where it
/// starts is held at that distance from the initial coordinates, whatever the step.
///
/// With [`SolveStrategy::Condensed`] the local unknowns of the plastic step are converged
/// and eliminated at each integration point, so the solver only sees the nodal
/// coordinates. With [`SolveStrategy::Monolithic`] they are unknowns of the solve too,
/// stepped together with the nodal coordinates through one sparse system, with or
/// without eliminating them; that needs a domain that supports it.
pub trait RateIndependentRoot<S, const D: usize> {
    fn root(
        &self,
        solver: impl RootFinding<NodalForcesSolid<D>, NodalStiffnessesSolid<D>, NodalCoordinates<D>>
        + RootFindingBlock<
            Vector,
            Vector,
            Vector,
            Vector,
            CscMatrix,
            CscMatrix,
            CscMatrix,
            CscMatrix,
        >,
        conditions: &impl ConstrainAt<<NodalCoordinates<D> as NodalState>::Layout>,
        applied_loads: &(
             impl LoadAt<NodalCoordinates<D>, NodalForcesSolid<D>>
             + LoadStiffnessAt<NodalCoordinates<D>, NodalStiffnessesSolid<D>>
         ),
        times: &[Quantity<Time>],
        strategy: SolveStrategy,
        abs_tol: Quantity,
    ) -> Result<(NodalCoordinatesHistory<D>, Vec<S>), OptimizationError>;
}

type Evaluation<const D: usize> = (
    NodalCoordinates<D>,
    NodalForcesSolid<D>,
    NodalStiffnessesSolid<D>,
);

/// What of a constraint the sparsity pattern of the sparse solver depends on: which
/// coordinates are eliminated, which are held by multipliers, or where the rows of the linear
/// constraint are nonzero, but not the values, which change from one load step to the next.
#[derive(PartialEq)]
enum ConstraintPart {
    Fixed(Vec<usize>),
    Multipliers(Vec<usize>),
    Linear(usize, usize, Vec<(usize, usize)>),
}

#[derive(PartialEq)]
struct ConstraintPattern(Vec<ConstraintPart>);

impl From<&[EqualityConstraint]> for ConstraintPattern {
    fn from(constraints: &[EqualityConstraint]) -> Self {
        Self(
            EqualityConstraint::merged(constraints.to_vec())
                .into_iter()
                .filter_map(|part| match part {
                    EqualityConstraint::None => None,
                    EqualityConstraint::Fixed(_) | EqualityConstraint::Prescribed(_) => {
                        let Some(indices) = EqualityConstraint::held_indices(&[part]) else {
                            unreachable!("A fixed or prescribed constraint holds indices.")
                        };
                        Some(ConstraintPart::Fixed(indices))
                    }
                    EqualityConstraint::Multipliers(entries) => Some(ConstraintPart::Multipliers(
                        entries.into_iter().map(|(index, _)| index).collect(),
                    )),
                    EqualityConstraint::Linear(matrix, _) => {
                        let mut nonzeros = Vec::new();
                        (0..matrix.len()).for_each(|row| {
                            (0..matrix.width()).for_each(|column| {
                                if matrix[row][column] != 0.0 {
                                    nonzeros.push((row, column))
                                }
                            })
                        });
                        Some(ConstraintPart::Linear(
                            matrix.len(),
                            matrix.width(),
                            nonzeros,
                        ))
                    }
                })
                .collect(),
        )
    }
}

/// The rows of the constraints for the monolithic solve, as the entries that are not zero.
///
/// What is fixed or prescribed is one row for each index, held at the guess when it is fixed
/// or prescribed at its initial value, so that the rows are never a dense matrix.
struct ConstraintRows {
    height: usize,
    width: usize,
    entries: BTreeMap<(usize, usize), f64>,
    values: Vec<f64>,
}

impl ConstraintRows {
    fn new(constraints: &[EqualityConstraint], guess: &Vector) -> Self {
        let mut rows = Self {
            height: 0,
            width: guess.len(),
            entries: BTreeMap::new(),
            values: Vec::new(),
        };
        let hold = |rows: &mut Self, index: usize, value: f64| {
            rows.entries.insert((rows.height, index), 1.0);
            rows.values.push(value);
            rows.height += 1;
        };
        for constraint in EqualityConstraint::merged(constraints.to_vec()) {
            match constraint {
                EqualityConstraint::Fixed(indices) => indices
                    .into_iter()
                    .for_each(|index| hold(&mut rows, index, guess[index])),
                EqualityConstraint::Prescribed(entries)
                | EqualityConstraint::Multipliers(entries) => {
                    entries
                        .into_iter()
                        .for_each(|(index, prescription)| match prescription {
                            Prescription::Initial => hold(&mut rows, index, guess[index]),
                            Prescription::Offset(offset) => {
                                hold(&mut rows, index, guess[index] + offset)
                            }
                            Prescription::Absolute(value) => hold(&mut rows, index, value),
                        })
                }
                EqualityConstraint::Linear(matrix, vector) => (0..matrix.len()).for_each(|row| {
                    (0..matrix.width()).for_each(|column| {
                        if matrix[row][column] != 0.0 {
                            rows.entries
                                .insert((rows.height, column), matrix[row][column]);
                        }
                    });
                    rows.values.push(vector[row]);
                    rows.height += 1;
                }),
                EqualityConstraint::None => {}
            }
        }
        rows
    }
    fn values(&self) -> Vector {
        let mut values = Vector::zero(self.height);
        self.values
            .iter()
            .enumerate()
            .for_each(|(row, &value)| values[row] = value);
        values
    }
}

/// The value built for a constraint, built again only when the pattern of the constraint
/// changes, since symbolic analysis of the sparse system is the same for every step of
/// the same pattern.
fn cached<'a, T>(
    cache: &'a mut Option<(ConstraintPattern, T)>,
    constraint: &[EqualityConstraint],
    build: impl FnOnce() -> T,
) -> &'a T {
    let pattern = ConstraintPattern::from(constraint);
    if cache.as_ref().is_none_or(|(cached, _)| *cached != pattern) {
        *cache = Some((pattern, build()));
    }
    &cache.as_ref().unwrap().1
}

impl<B, S, const D: usize> RateIndependentRoot<S, D> for Model<B, D>
where
    B: RateIndependentElements<S, D>,
    S: Clone,
{
    fn root(
        &self,
        solver: impl RootFinding<NodalForcesSolid<D>, NodalStiffnessesSolid<D>, NodalCoordinates<D>>
        + RootFindingBlock<
            Vector,
            Vector,
            Vector,
            Vector,
            CscMatrix,
            CscMatrix,
            CscMatrix,
            CscMatrix,
        >,
        conditions: &impl ConstrainAt<<NodalCoordinates<D> as NodalState>::Layout>,
        applied_loads: &(
             impl LoadAt<NodalCoordinates<D>, NodalForcesSolid<D>>
             + LoadStiffnessAt<NodalCoordinates<D>, NodalStiffnessesSolid<D>>
         ),
        times: &[Quantity<Time>],
        strategy: SolveStrategy,
        abs_tol: Quantity,
    ) -> Result<(NodalCoordinatesHistory<D>, Vec<S>), OptimizationError> {
        let nodal_coordinates: NodalCoordinates<D> = self.coordinates().clone().into();
        let state = self.blocks().initial_state();
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
        let mut condensed_cache = None;
        let mut monolithic_cache = None;
        let anchor = nodal_coordinates.clone();
        let mut advance = |time: Quantity<Time>,
                           start: &NodalCoordinates<D>,
                           start_state: &S|
         -> Result<(NodalCoordinates<D>, S), OptimizationError> {
            let mut nodal_coordinates = start.clone();
            let mut state = start_state.clone();
            let constraint =
                EqualityConstraint::from_reference(conditions.constraint_at(time), &anchor);
            let frozen_state = state.clone();
            if let (Some(monolithic), SolveStrategy::Monolithic { elimination }) =
                (system.take(), &strategy)
            {
                let (num_global, num_local) = (monolithic.num_global(), monolithic.num_local());
                let mut initial = Vector::zero(num_global);
                nodal_coordinates.fill_into(&mut initial);
                let rows = ConstraintRows::new(&constraint, &initial);
                let (sparse, mut constraint_matrix) =
                    cached(&mut monolithic_cache, &constraint, || {
                        let mut pattern = if *elimination {
                            monolithic.tangent_uu.pattern().to_vec()
                        } else {
                            monolithic.pattern(rows.height)
                        };
                        let mut constraint_pattern = Vec::new();
                        rows.entries.keys().for_each(|&(row, column)| {
                            constraint_pattern.push((row, column));
                            pattern.push((num_global + row, column));
                            pattern.push((column, num_global + row))
                        });
                        pattern.sort_unstable();
                        pattern.dedup();
                        (
                            SparseSolver::from_pattern(
                                num_global + rows.height + if *elimination { 0 } else { num_local },
                                pattern,
                                false,
                            ),
                            CscMatrix::from_pattern(rows.height, rows.width, constraint_pattern),
                        )
                    })
                    .clone();
                constraint_matrix
                    .fill(|row, column| rows.entries.get(&(row, column)).copied().unwrap_or(0.0));
                let cache = RefCell::<Option<(Vector, Vector)>>::new(None);
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
                let solved = solver.root_block(
                    |global: &Vector, local: &Vector| {
                        let mut system = cell.borrow_mut();
                        evaluate(global, local, &mut system)?;
                        let mut loaded = NodalForcesSolid::<D>::zero(global.len() / D);
                        applied_loads.subtract_at(
                            time,
                            &NodalCoordinates::from(global.clone()),
                            &mut loaded,
                        )?;
                        let mut flat = Vector::zero(global.len());
                        loaded.fill_into(&mut flat);
                        Ok(&system.residual_global + &flat)
                    },
                    |global: &Vector, local: &Vector| {
                        let mut system = cell.borrow_mut();
                        evaluate(global, local, &mut system)?;
                        Ok(system.residual_local.clone())
                    },
                    |global: &Vector, local: &Vector| {
                        let mut system = cell.borrow_mut();
                        evaluate(global, local, &mut system)?;
                        let mut tangent_uu = system.tangent_uu.clone();
                        let mut loaded = NodalStiffnessesSolid::<D>::zero(global.len() / D);
                        applied_loads.add_stiffness_at(
                            time,
                            &NodalCoordinates::from(global.clone()),
                            &mut loaded,
                        )?;
                        loaded.iter().enumerate().for_each(|(a, row)| {
                            row.entries().for_each(|(b, block)| {
                                (0..D).for_each(|i| {
                                    (0..D).for_each(|j| {
                                        tangent_uu.accumulate(
                                            D * a + i,
                                            D * b + j,
                                            block[i][j].value(),
                                        )
                                    })
                                })
                            })
                        });
                        Ok((
                            tangent_uu,
                            system.tangent_vu.clone(),
                            system.tangent_uv.clone(),
                            system.tangent_vv.clone(),
                        ))
                    },
                    (initial, Vector::zero(num_local)),
                    (constraint_matrix, rows.values()),
                    (
                        CscMatrix::from_pattern(0, num_local, Vec::new()),
                        Vector::zero(0),
                    ),
                    Some(sparse),
                    strategy.clone(),
                );
                system = Some(cell.into_inner());
                let (new_global, new_local) = solved?;
                nodal_coordinates = NodalCoordinates::from(new_global);
                state = self
                    .blocks()
                    .monolithic_state(&frozen_state, &new_local)
                    .map_err(|error| error.to_string())?;
            } else if let SolveStrategy::Condensed(local_solver) = &strategy {
                let sparse = cached(&mut condensed_cache, &constraint, || {
                    solver_from_neighbors(&neighbors, &constraint, D, false)
                })
                .clone();
                let cache = RefCell::<Option<Evaluation<D>>>::new(None);
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
                        let mut nodal_forces = cache.borrow().as_ref().unwrap().1.clone();
                        applied_loads.subtract_at(time, coordinates, &mut nodal_forces)?;
                        Ok(nodal_forces)
                    },
                    |coordinates: &NodalCoordinates<D>| {
                        evaluate(coordinates)?;
                        let mut stiffnesses = cache.borrow().as_ref().unwrap().2.clone();
                        applied_loads.add_stiffness_at(time, coordinates, &mut stiffnesses)?;
                        Ok(stiffnesses)
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
            Ok((nodal_coordinates, state))
        };
        let reported = adapt(
            times,
            abs_tol,
            (nodal_coordinates, state),
            &mut advance,
            |coordinates, updated_coordinates, state, updated_state| {
                self.blocks()
                    .step_error(coordinates, updated_coordinates, state, updated_state)
                    .map_err(|error| OptimizationError::from(error.to_string()))
            },
            OptimizationError::from,
        )?;
        let mut coordinates_history = NodalCoordinatesHistory::new();
        let mut state_history = Vec::new();
        for (_, coordinates, state) in reported {
            coordinates_history.push(coordinates);
            state_history.push(state);
        }
        Ok((coordinates_history, state_history))
    }
}
