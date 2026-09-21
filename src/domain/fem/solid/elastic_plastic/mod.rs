use crate::{
    fem::{
        ElementModel, Model, NodalCoordinates, NodalCoordinatesHistory,
        block::solid::elastic_plastic::{
            MonolithicElasticPlasticElements, MonolithicSystem, PlasticStateVariablesField,
        },
    },
    math::{
        Jacobian, Tensor, TensorVec, Vector,
        optimize::{
            EqualityConstraint, FirstOrderRootFindingBlock, OptimizationError, SolveStrategy,
        },
        sparse::{CscMatrix, SparseSolver},
    },
};
use std::cell::RefCell;

pub use crate::domain::solid::elastic_plastic::{ElasticPlasticElements, ElasticPlasticRoot};

/// Monolithic (block) root-finding for elastic-plastic solid finite elements: the
/// local unknowns of the return map at every integration point are unknowns of the
/// outer solve, stepped together with the nodal coordinates through one sparse system.
///
/// [`SolveStrategy::Condensed`] is [`ElasticPlasticRoot`]. Monolithic, with or without
/// eliminating the local unknowns, and only [`EqualityConstraint::Linear`] boundary
/// conditions are supported.
pub trait FirstOrderRootBlock<const G: usize> {
    fn root(
        &self,
        solver: impl FirstOrderRootFindingBlock<
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
    ) -> Result<
        (
            NodalCoordinatesHistory<3>,
            Vec<PlasticStateVariablesField<G>>,
        ),
        OptimizationError,
    >;
}

impl<B, const G: usize> FirstOrderRootBlock<G> for Model<B, 3>
where
    B: MonolithicElasticPlasticElements<G>
        + ElasticPlasticElements<PlasticStateVariablesField<G>, 3>,
{
    fn root(
        &self,
        solver: impl FirstOrderRootFindingBlock<
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
    ) -> Result<
        (
            NodalCoordinatesHistory<3>,
            Vec<PlasticStateVariablesField<G>>,
        ),
        OptimizationError,
    > {
        assert!(
            matches!(strategy, SolveStrategy::Monolithic { .. }),
            "FirstOrderRootBlock only supports SolveStrategy::Monolithic; \
             SolveStrategy::Condensed is what ElasticPlasticRoot does"
        );
        let eliminating = matches!(strategy, SolveStrategy::Monolithic { elimination: true });
        let mut nodal_coordinates: NodalCoordinates<3> = self.coordinates().clone().into();
        let mut state = self.blocks().initial_state();
        let mut coordinates_history = NodalCoordinatesHistory::new();
        let mut state_history = Vec::new();
        coordinates_history.push(nodal_coordinates.clone());
        state_history.push(state.clone());
        let num_nodes = nodal_coordinates.len();
        let mut system = self.blocks().monolithic_system(num_nodes);
        let (num_global, num_local) = (system.num_global(), system.num_local());
        for constraint in boundary_conditions {
            let EqualityConstraint::Linear(matrix, vector) = constraint else {
                panic!("FirstOrderRootBlock only supports EqualityConstraint::Linear")
            };
            //
            // Eliminating the local unknowns leaves the sparse solver the global system
            // alone, whose pattern is that of the stiffness.
            //
            let mut pattern = if eliminating {
                system.tangent_uu.pattern().to_vec()
            } else {
                system.pattern(matrix.len())
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
                num_global + matrix.len() + if eliminating { 0 } else { num_local },
                pattern,
                false,
            );
            let mut constraint_matrix =
                CscMatrix::from_pattern(matrix.len(), matrix.width(), constraint_pattern);
            constraint_matrix.fill(|row, column| matrix[row][column]);
            let frozen_state = state.clone();
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
            let cell = RefCell::new(system);
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
            system = cell.into_inner();
            coordinates_history.push(nodal_coordinates.clone());
            state_history.push(state.clone());
        }
        Ok((coordinates_history, state_history))
    }
}
