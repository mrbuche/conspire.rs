use crate::{
    fem::{
        ElementModel, ElementModelError, Elements, Model, NodalCoordinates,
        NodalCoordinatesHistory,
        block::{
            finalize_node_neighbors,
            solid::elastic_plastic::{
                MonolithicElasticPlasticElements, PlasticStateVariablesField,
            },
            solver_from_neighbors,
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::{
        Jacobian, Matrix, Tensor, TensorVec, Vector,
        optimize::{
            EqualityConstraint, FirstOrderRootFinding, FirstOrderRootFindingBlock,
            OptimizationError, SolveStrategy,
        },
        sparse::CscMatrix,
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

/// Monolithic (block) root-finding for elastic-plastic solid finite elements, with the
/// plastic multiplier field a free unknown of the outer solve rather than condensed out
/// at each integration point by [`ElasticPlasticRoot`].
///
/// With [`SolveStrategy::Condensed`], this delegates straight to [`ElasticPlasticRoot`]:
/// the two compute the identical residual and tangent (both converge the plastic
/// multiplier per quadrature point and form the same analytically eliminated tangent),
/// so going through the block solver's dense `K_uv`/`K_vu` assembly would only add cost
/// -- and, unlike at a single material point, that cost is paid at every quadrature
/// point of every element, so it is not just slower but scales as
/// `O(num_dofs^2 * num_quadrature_points)` instead of linearly. [`SolveStrategy::Monolithic`]
/// genuinely needs the block solver and is only tractable on small meshes as a result;
/// only [`EqualityConstraint::Linear`] boundary conditions are supported.
pub trait FirstOrderRootBlock<const G: usize, const D: usize> {
    fn root(
        &self,
        solver: impl FirstOrderRootFindingBlock<
            Vector,
            Vector,
            NodalForcesSolid<D>,
            Vector,
            Matrix,
            Matrix,
            Matrix,
            Matrix,
        > + FirstOrderRootFinding<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        boundary_conditions: &[EqualityConstraint],
        strategy: SolveStrategy,
    ) -> Result<
        (
            NodalCoordinatesHistory<D>,
            Vec<PlasticStateVariablesField<G>>,
        ),
        OptimizationError,
    >;
}

impl<B, const G: usize> FirstOrderRootBlock<G, 3> for Model<B, 3>
where
    B: MonolithicElasticPlasticElements<G, 3>
        + ElasticPlasticElements<PlasticStateVariablesField<G>, 3>,
{
    fn root(
        &self,
        solver: impl FirstOrderRootFindingBlock<
            Vector,
            Vector,
            NodalForcesSolid<3>,
            Vector,
            Matrix,
            Matrix,
            Matrix,
            Matrix,
        > + FirstOrderRootFinding<
            NodalForcesSolid<3>,
            NodalStiffnessesSolid<3>,
            NodalCoordinates<3>,
        >,
        boundary_conditions: &[EqualityConstraint],
        strategy: SolveStrategy,
    ) -> Result<
        (
            NodalCoordinatesHistory<3>,
            Vec<PlasticStateVariablesField<G>>,
        ),
        OptimizationError,
    > {
        if let SolveStrategy::Condensed(_) = strategy {
            return ElasticPlasticRoot::root(self, solver, boundary_conditions);
        }
        let mut nodal_coordinates: NodalCoordinates<3> = self.coordinates().clone().into();
        let mut state = self.blocks().initial_state();
        let mut coordinates_history = NodalCoordinatesHistory::new();
        let mut state_history = Vec::new();
        coordinates_history.push(nodal_coordinates.clone());
        state_history.push(state.clone());
        let num_global = 3 * nodal_coordinates.len();
        let num_local = self.blocks().num_local();
        for constraint in boundary_conditions {
            let (matrix, vector) = match constraint {
                EqualityConstraint::Linear(matrix, vector) => (matrix, vector),
                _ => panic!(
                    "FirstOrderRootBlock only supports EqualityConstraint::Linear boundary conditions"
                ),
            };
            let mut global_pattern = Vec::new();
            (0..matrix.len()).for_each(|row| {
                (0..matrix.width()).for_each(|column| {
                    if matrix[row][column] != 0.0 {
                        global_pattern.push((row, column));
                    }
                })
            });
            let mut global_matrix =
                CscMatrix::from_pattern(matrix.len(), matrix.width(), global_pattern);
            global_matrix.fill(|_, _| 1.0);
            let frozen_state = state.clone();
            let mut initial_u = Vector::zero(num_global);
            nodal_coordinates.fill_into(&mut initial_u);
            let (new_u, _) = solver.root_block(
                |u: &Vector, multipliers: &Vector| {
                    self.blocks()
                        .monolithic_contributions(
                            &NodalCoordinates::from(u.clone()),
                            &frozen_state,
                            multipliers,
                        )
                        .map(|(residual_global, ..)| residual_global)
                        .map_err(|error| error.to_string())
                },
                |u: &Vector, multipliers: &Vector| {
                    self.blocks()
                        .monolithic_contributions(
                            &NodalCoordinates::from(u.clone()),
                            &frozen_state,
                            multipliers,
                        )
                        .map(|(_, residual_local, ..)| residual_local)
                        .map_err(|error| error.to_string())
                },
                |u: &Vector, multipliers: &Vector| {
                    self.blocks()
                        .monolithic_contributions(
                            &NodalCoordinates::from(u.clone()),
                            &frozen_state,
                            multipliers,
                        )
                        .map(|(_, _, k_uu, k_uv, k_vu, k_vv)| (k_uu, k_vu, k_uv, k_vv))
                        .map_err(|error| error.to_string())
                },
                (initial_u, Vector::zero(num_local)),
                (global_matrix, vector.clone()),
                (
                    CscMatrix::from_pattern(0, num_local, Vec::new()),
                    Vector::zero(0),
                ),
                None,
                strategy.clone(),
            )?;
            nodal_coordinates = NodalCoordinates::from(new_u);
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
