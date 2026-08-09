use crate::{
    fem::{
        ElementModel, ElementModelError, Elements, Model, NodalCoordinates,
        block::{finalize_node_neighbors, solver_from_neighbors},
        solid::{
            NodalForcesSolid, NodalStiffnessesSolid,
            elastic::internal_variables::{ElasticIVElements, InternalVariablesField},
        },
    },
    math::{
        Scalar, Tensor, Vector,
        optimize::{
            EqualityConstraint, OptimizationError, SecondOrderOptimization,
            SecondOrderOptimizationIncremental, SolveStrategy,
        },
    },
};
use std::cell::RefCell;

pub trait HyperelasticIVElements<const G: usize, V, const D: usize>
where
    Self: ElasticIVElements<G, V, D>,
    V: Tensor,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        internal_variables: &InternalVariablesField<G, V>,
    ) -> Result<Scalar, ElementModelError>;
}

impl<B, const G: usize, V, const D: usize> HyperelasticIVElements<G, V, D> for Model<B, D>
where
    B: HyperelasticIVElements<G, V, D>,
    V: Tensor,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        internal_variables: &InternalVariablesField<G, V>,
    ) -> Result<Scalar, ElementModelError> {
        self.blocks
            .helmholtz_free_energy(nodal_coordinates, internal_variables)
    }
}

/// Second-order minimization for hyperelastic models whose internal variables
/// are condensed out at every integration point.
pub trait SecondOrderMinimizeIV<const G: usize, V, const D: usize>
where
    V: Tensor,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        > + SecondOrderOptimizationIncremental<
            Scalar,
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        strategy: SolveStrategy,
    ) -> Result<NodalCoordinates<D>, OptimizationError>;
}

impl<B, const G: usize, V, const D: usize> SecondOrderMinimizeIV<G, V, D> for Model<B, D>
where
    B: HyperelasticIVElements<G, V, D>,
    V: Tensor,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        > + SecondOrderOptimizationIncremental<
            Scalar,
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        strategy: SolveStrategy,
    ) -> Result<NodalCoordinates<D>, OptimizationError> {
        let initial = self.internal_variables_initial();
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        //
        // Condensation preserves symmetry: the local block and the couplings
        // are those of one energy Hessian, so the Schur complement is symmetric
        // wherever the unreduced tangent was.
        //
        let sparse = solver_from_neighbors(&neighbors, &equality_constraint, D, true);
        match strategy {
            //
            // The internal variables are solved before they are used, so the
            // energy is one of the nodal coordinates alone, the solver being
            // free to evaluate wherever it likes.
            //
            SolveStrategy::Condensed(ref local_solver) => {
                let cache: RefCell<Option<(NodalCoordinates<D>, InternalVariablesField<G, V>)>> =
                    RefCell::new(None);
                let solved = |nodal_coordinates: &NodalCoordinates<D>| {
                    if let Some((ref at, ref variables)) = *cache.borrow()
                        && at == nodal_coordinates
                    {
                        return Ok(variables.clone());
                    }
                    let warm = match *cache.borrow() {
                        Some((_, ref variables)) => variables.clone(),
                        None => initial.clone(),
                    };
                    let variables =
                        self.internal_variables_root(local_solver, nodal_coordinates, &warm)?;
                    *cache.borrow_mut() = Some((nodal_coordinates.clone(), variables.clone()));
                    Ok::<_, ElementModelError>(variables)
                };
                solver.minimize(
                    |nodal_coordinates: &NodalCoordinates<D>| {
                        Ok(self.helmholtz_free_energy(
                            nodal_coordinates,
                            &solved(nodal_coordinates)?,
                        )?)
                    },
                    |nodal_coordinates: &NodalCoordinates<D>| {
                        Ok(self.nodal_forces(nodal_coordinates, &solved(nodal_coordinates)?)?)
                    },
                    |nodal_coordinates: &NodalCoordinates<D>| {
                        Ok(self.nodal_stiffnesses(nodal_coordinates, &solved(nodal_coordinates)?)?)
                    },
                    self.coordinates().clone().into(),
                    equality_constraint,
                    Some(sparse),
                )
            }
            //
            // The internal variables are carried instead, stepped by their
            // share of whatever step the nodal coordinates take. What the line
            // search weighs is the energy of the whole state, so they are moved
            // to where a step would put them before the energy there is asked
            // for, and left there only if that step is the one taken.
            //
            SolveStrategy::Monolithic { elimination: true } => {
                let committed = RefCell::new(initial.clone());
                let internal_variables = RefCell::new(initial);
                solver.minimize_incremental(
                    |nodal_coordinates: &NodalCoordinates<D>| {
                        Ok(self.helmholtz_free_energy(
                            nodal_coordinates,
                            &internal_variables.borrow(),
                        )?)
                    },
                    |nodal_coordinates: &NodalCoordinates<D>| {
                        Ok(self.nodal_forces_eliminated(
                            nodal_coordinates,
                            &internal_variables.borrow(),
                        )?)
                    },
                    |nodal_coordinates: &NodalCoordinates<D>| {
                        Ok(self
                            .nodal_stiffnesses(nodal_coordinates, &internal_variables.borrow())?)
                    },
                    |nodal_coordinates: &NodalCoordinates<D>,
                     decrement: &Vector,
                     step: Scalar,
                     commit: bool| {
                        let stepped = self.internal_variables_increment(
                            nodal_coordinates,
                            &committed.borrow(),
                            &NodalCoordinates::from(decrement.clone()),
                            step,
                        )?;
                        if commit {
                            *committed.borrow_mut() = stepped.clone()
                        }
                        *internal_variables.borrow_mut() = stepped;
                        Ok(())
                    },
                    self.coordinates().clone().into(),
                    equality_constraint,
                    Some(sparse),
                )
            }
            SolveStrategy::Monolithic { elimination: false } => unimplemented!(
                "The internal variables must be unknowns of the solver to be solved with it."
            ),
        }
    }
}
