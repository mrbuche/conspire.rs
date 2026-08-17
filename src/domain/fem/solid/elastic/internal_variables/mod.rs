use crate::{
    fem::{
        ElementModel, ElementModelError, Elements, Model, NodalCoordinates,
        block::{
            element::solid::elastic::internal_variables::InternalVariables,
            finalize_node_neighbors, solver_from_neighbors,
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::{
        Scalar, Tensor, TensorVector, Vector,
        optimize::{
            EqualityConstraint, FirstOrderRootFinding, FirstOrderRootFindingIncremental,
            NewtonRaphson, OptimizationError, SolveStrategy,
        },
    },
};
use std::cell::{Ref, RefCell};

/// The internal variables held at every integration point of every element.
pub type InternalVariablesField<const G: usize, V> = TensorVector<InternalVariables<G, V>>;

pub trait ElasticIVElements<const G: usize, V, const D: usize>
where
    Self: Elements,
    V: Tensor,
{
    /// The internal variables every integration point starts from.
    fn internal_variables_initial(&self) -> InternalVariablesField<G, V>;
    /// Steps the internal variables everywhere alongside a decrement of the
    /// nodal coordinates, rather than solving them where the coordinates now
    /// are.
    ///
    /// This is the second row of the Newton system the solver never assembles,
    /// each integration point taking its own share of the increment.
    fn internal_variables_increment(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        internal_variables: &InternalVariablesField<G, V>,
        nodal_decrement: &NodalCoordinates<D>,
        step: Scalar,
    ) -> Result<InternalVariablesField<G, V>, ElementModelError>;
    /// Solves the internal variables everywhere, holding the deformation fixed.
    ///
    /// Every integration point is independent of every other, so this is a
    /// gather of local solves rather than a system.
    fn internal_variables_root(
        &self,
        local_solver: &NewtonRaphson,
        nodal_coordinates: &NodalCoordinates<D>,
        internal_variables: &InternalVariablesField<G, V>,
    ) -> Result<InternalVariablesField<G, V>, ElementModelError>;
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        internal_variables: &InternalVariablesField<G, V>,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError>;
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        internal_variables: &InternalVariablesField<G, V>,
    ) -> Result<NodalForcesSolid<D>, ElementModelError> {
        let mut nodal_forces = NodalForcesSolid::zero(nodal_coordinates.len());
        self.nodal_forces_into(nodal_coordinates, internal_variables, &mut nodal_forces)?;
        Ok(nodal_forces)
    }
    /// The nodal forces with the residual of the internal variables eliminated
    /// into them, for when they are carried rather than solved.
    ///
    /// ```math
    /// \mathbf{r}_u - \mathcal{K}_{uv}\mathcal{K}_{vv}^{-1}\mathbf{r}_v
    /// ```
    fn nodal_forces_eliminated_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        internal_variables: &InternalVariablesField<G, V>,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError>;
    fn nodal_forces_eliminated(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        internal_variables: &InternalVariablesField<G, V>,
    ) -> Result<NodalForcesSolid<D>, ElementModelError> {
        let mut nodal_forces = NodalForcesSolid::zero(nodal_coordinates.len());
        self.nodal_forces_eliminated_into(
            nodal_coordinates,
            internal_variables,
            &mut nodal_forces,
        )?;
        Ok(nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        internal_variables: &InternalVariablesField<G, V>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        internal_variables: &InternalVariablesField<G, V>,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError> {
        let mut nodal_stiffnesses = NodalStiffnessesSolid::zero(nodal_coordinates.len());
        self.nodal_stiffnesses_into(
            nodal_coordinates,
            internal_variables,
            &mut nodal_stiffnesses,
        )?;
        Ok(nodal_stiffnesses)
    }
}

impl<B, const G: usize, V, const D: usize> ElasticIVElements<G, V, D> for Model<B, D>
where
    B: ElasticIVElements<G, V, D>,
    V: Tensor,
{
    fn internal_variables_initial(&self) -> InternalVariablesField<G, V> {
        self.blocks.internal_variables_initial()
    }
    fn internal_variables_increment(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        internal_variables: &InternalVariablesField<G, V>,
        nodal_decrement: &NodalCoordinates<D>,
        step: Scalar,
    ) -> Result<InternalVariablesField<G, V>, ElementModelError> {
        self.blocks.internal_variables_increment(
            nodal_coordinates,
            internal_variables,
            nodal_decrement,
            step,
        )
    }
    fn internal_variables_root(
        &self,
        local_solver: &NewtonRaphson,
        nodal_coordinates: &NodalCoordinates<D>,
        internal_variables: &InternalVariablesField<G, V>,
    ) -> Result<InternalVariablesField<G, V>, ElementModelError> {
        self.blocks
            .internal_variables_root(local_solver, nodal_coordinates, internal_variables)
    }
    fn nodal_forces_eliminated_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        internal_variables: &InternalVariablesField<G, V>,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.blocks.nodal_forces_eliminated_into(
            nodal_coordinates,
            internal_variables,
            nodal_forces,
        )
    }
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        internal_variables: &InternalVariablesField<G, V>,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_forces_into(nodal_coordinates, internal_variables, nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        internal_variables: &InternalVariablesField<G, V>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_stiffnesses_into(nodal_coordinates, internal_variables, nodal_stiffnesses)
    }
}

/// The internal variables as a function of the nodal coordinates, solved
/// wherever they are asked for.
///
/// The residual and the tangent are asked for at the same coordinates, so the
/// solve is remembered rather than repeated, and what it converged to last is
/// where the next one starts.
pub struct SolvedInternalVariables<'a, B, const G: usize, V, const D: usize>
where
    V: Tensor,
{
    initial: InternalVariablesField<G, V>,
    local_solver: &'a NewtonRaphson,
    model: &'a Model<B, D>,
    solved: RefCell<Option<(NodalCoordinates<D>, InternalVariablesField<G, V>)>>,
}

impl<'a, B, const G: usize, V, const D: usize> SolvedInternalVariables<'a, B, G, V, D>
where
    B: ElasticIVElements<G, V, D>,
    V: Tensor,
{
    pub fn new(
        model: &'a Model<B, D>,
        local_solver: &'a NewtonRaphson,
        initial: InternalVariablesField<G, V>,
    ) -> Self {
        Self {
            initial,
            local_solver,
            model,
            solved: RefCell::new(None),
        }
    }
    pub fn at(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<InternalVariablesField<G, V>, ElementModelError> {
        if let Some((ref at, ref variables)) = *self.solved.borrow()
            && at == nodal_coordinates
        {
            return Ok(variables.clone());
        }
        let warm = match *self.solved.borrow() {
            Some((_, ref variables)) => variables.clone(),
            None => self.initial.clone(),
        };
        let variables =
            self.model
                .internal_variables_root(self.local_solver, nodal_coordinates, &warm)?;
        *self.solved.borrow_mut() = Some((nodal_coordinates.clone(), variables.clone()));
        Ok(variables)
    }
}

/// The internal variables carried alongside the nodal coordinates, stepped by
/// their share of whatever increment the solver lends out.
///
/// A step is offered before it is taken, so what the increment is measured from
/// and what the residual is evaluated at come apart while one is being
/// considered. Increments are always taken from the state last kept, which is
/// what makes offering several of them in a row harmless.
pub struct CarriedInternalVariables<'a, B, const G: usize, V, const D: usize>
where
    V: Tensor,
{
    committed: RefCell<InternalVariablesField<G, V>>,
    model: &'a Model<B, D>,
    stepped: RefCell<InternalVariablesField<G, V>>,
}

impl<'a, B, const G: usize, V, const D: usize> CarriedInternalVariables<'a, B, G, V, D>
where
    B: ElasticIVElements<G, V, D>,
    V: Tensor,
{
    pub fn new(model: &'a Model<B, D>, initial: InternalVariablesField<G, V>) -> Self {
        Self {
            committed: RefCell::new(initial.clone()),
            model,
            stepped: RefCell::new(initial),
        }
    }
    pub fn stepped(&self) -> Ref<'_, InternalVariablesField<G, V>> {
        self.stepped.borrow()
    }
    pub fn step(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_decrement: &Vector,
        step: Scalar,
        commit: bool,
    ) -> Result<(), ElementModelError> {
        let stepped = self.model.internal_variables_increment(
            nodal_coordinates,
            &self.committed.borrow(),
            &NodalCoordinates::from(nodal_decrement.clone()),
            step,
        )?;
        if commit {
            *self.committed.borrow_mut() = stepped.clone()
        }
        *self.stepped.borrow_mut() = stepped;
        Ok(())
    }
}

/// First-order root-finding for elastic models whose internal variables are
/// condensed out at every integration point.
pub trait FirstOrderRootIV<const G: usize, V, const D: usize>
where
    V: Tensor,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        > + FirstOrderRootFindingIncremental<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        strategy: SolveStrategy,
    ) -> Result<NodalCoordinates<D>, OptimizationError>;
}

impl<B, const G: usize, V, const D: usize> FirstOrderRootIV<G, V, D> for Model<B, D>
where
    B: ElasticIVElements<G, V, D>,
    V: Tensor,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        > + FirstOrderRootFindingIncremental<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        strategy: SolveStrategy,
    ) -> Result<NodalCoordinates<D>, OptimizationError> {
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        //
        // Either way the solver only ever sees the nodal coordinates, so the
        // sparsity is that of an ordinary mesh. The internal variables are
        // element-local, so what they contribute to the tangent lands where the
        // nodes of their own element already meet.
        //
        let sparse = solver_from_neighbors(&neighbors, &equality_constraint, D, false);
        let initial = self.internal_variables_initial();
        match strategy {
            //
            // The internal variables are solved before they are used, so the
            // residual is one of the nodal coordinates alone, the solver being
            // free to evaluate wherever it likes.
            //
            SolveStrategy::Condensed(ref local_solver) => {
                let solved = SolvedInternalVariables::new(self, local_solver, initial);
                solver.root(
                    |nodal_coordinates: &NodalCoordinates<D>| {
                        Ok(self.nodal_forces(nodal_coordinates, &solved.at(nodal_coordinates)?)?)
                    },
                    |nodal_coordinates: &NodalCoordinates<D>| {
                        Ok(self
                            .nodal_stiffnesses(nodal_coordinates, &solved.at(nodal_coordinates)?)?)
                    },
                    self.coordinates().clone().into(),
                    equality_constraint,
                    Some(sparse),
                )
            }
            //
            // The internal variables are carried instead, stepped once per
            // iteration by the increment the solver lends out. They are never
            // at their own root along the way, so their residual has to be
            // eliminated into the one the solver does see.
            //
            SolveStrategy::Monolithic { elimination: true } => {
                let carried = CarriedInternalVariables::new(self, initial);
                solver.root_incremental(
                    |nodal_coordinates: &NodalCoordinates<D>| {
                        Ok(self.nodal_forces_eliminated(nodal_coordinates, &carried.stepped())?)
                    },
                    |nodal_coordinates: &NodalCoordinates<D>| {
                        Ok(self.nodal_stiffnesses(nodal_coordinates, &carried.stepped())?)
                    },
                    |nodal_coordinates: &NodalCoordinates<D>,
                     decrement: &Vector,
                     step: Scalar,
                     commit: bool| {
                        Ok(carried.step(nodal_coordinates, decrement, step, commit)?)
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
