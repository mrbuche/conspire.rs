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
        Tensor, TensorVector, Vector,
        optimize::{
            EqualityConstraint, FirstOrderRootFinding, FirstOrderRootFindingIncremental,
            NewtonRaphson, OptimizationError, SolveStrategy,
        },
    },
};
use std::cell::RefCell;

/// The internal variables held at every integration point of every element.
pub type InternalVariablesField<const G: usize, V> = TensorVector<InternalVariables<G, V>>;

pub trait ElasticIVElements<const G: usize, V, T1, T2, T3, const D: usize>
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

impl<B, const G: usize, V, T1, T2, T3, const D: usize> ElasticIVElements<G, V, T1, T2, T3, D>
    for Model<B, D>
where
    B: ElasticIVElements<G, V, T1, T2, T3, D>,
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
    ) -> Result<InternalVariablesField<G, V>, ElementModelError> {
        self.blocks.internal_variables_increment(
            nodal_coordinates,
            internal_variables,
            nodal_decrement,
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

/// First-order root-finding for elastic models whose internal variables are
/// condensed out at every integration point.
pub trait FirstOrderRootIV<const G: usize, V, T1, T2, T3, J, H, X, const D: usize>
where
    V: Tensor,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderRootFinding<J, H, X> + FirstOrderRootFindingIncremental<J, H, X>,
        strategy: SolveStrategy,
    ) -> Result<X, OptimizationError>;
}

impl<B, const G: usize, V, T1, T2, T3, const D: usize>
    FirstOrderRootIV<
        G,
        V,
        T1,
        T2,
        T3,
        NodalForcesSolid<D>,
        NodalStiffnessesSolid<D>,
        NodalCoordinates<D>,
        D,
    > for Model<B, D>
where
    B: ElasticIVElements<G, V, T1, T2, T3, D>,
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
            // The residual and the tangent are asked for at the same
            // coordinates, so the solve is remembered rather than repeated, and
            // what it converged to last is where the next one starts.
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
                solver.root(
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
            // The internal variables are carried instead, stepped once per
            // iteration by the increment the solver lends out. They are never
            // at their own root along the way, so their residual has to be
            // eliminated into the one the solver does see.
            //
            SolveStrategy::Monolithic { elimination: true } => {
                let internal_variables = RefCell::new(initial);
                solver.root_incremental(
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
                    |nodal_coordinates: &NodalCoordinates<D>, decrement: &Vector| {
                        let stepped = self.internal_variables_increment(
                            nodal_coordinates,
                            &internal_variables.borrow(),
                            &NodalCoordinates::from(decrement.clone()),
                        )?;
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
