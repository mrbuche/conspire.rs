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
        Tensor, TensorVector,
        optimize::{EqualityConstraint, FirstOrderRootFinding, OptimizationError, SolveStrategy},
    },
};

/// The internal variables held at every integration point of every element.
pub type InternalVariablesField<const G: usize, V> = TensorVector<InternalVariables<G, V>>;

pub trait ElasticIVElements<const G: usize, V, T1, T2, T3, const D: usize>
where
    Self: Elements,
    V: Tensor,
{
    /// The internal variables every integration point starts from.
    fn internal_variables_initial(&self) -> InternalVariablesField<G, V>;
    /// Solves the internal variables everywhere, holding the deformation fixed.
    ///
    /// Every integration point is independent of every other, so this is a
    /// gather of local solves rather than a system.
    fn internal_variables_root(
        &self,
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
    fn internal_variables_root(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        internal_variables: &InternalVariablesField<G, V>,
    ) -> Result<InternalVariablesField<G, V>, ElementModelError> {
        self.blocks
            .internal_variables_root(nodal_coordinates, internal_variables)
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
        solver: impl FirstOrderRootFinding<J, H, X>,
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
        >,
        strategy: SolveStrategy,
    ) -> Result<NodalCoordinates<D>, OptimizationError> {
        match strategy {
            SolveStrategy::Condensed => {}
            SolveStrategy::Monolithic { .. } => unimplemented!(
                "The internal variables must be unknowns of the solver to be solved with it."
            ),
        }
        //
        // The internal variables of an integration point are solved before they
        // are used, so the residual is one of the nodal coordinates alone. They
        // are solved afresh each time rather than carried, the solver being free
        // to evaluate wherever it likes.
        //
        let initial = self.internal_variables_initial();
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let sparse = solver_from_neighbors(&neighbors, &equality_constraint, D, false);
        solver.root(
            |nodal_coordinates: &NodalCoordinates<D>| {
                let internal_variables =
                    self.internal_variables_root(nodal_coordinates, &initial)?;
                Ok(self.nodal_forces(nodal_coordinates, &internal_variables)?)
            },
            |nodal_coordinates: &NodalCoordinates<D>| {
                let internal_variables =
                    self.internal_variables_root(nodal_coordinates, &initial)?;
                Ok(self.nodal_stiffnesses(nodal_coordinates, &internal_variables)?)
            },
            self.coordinates().clone().into(),
            equality_constraint,
            Some(sparse),
        )
    }
}
