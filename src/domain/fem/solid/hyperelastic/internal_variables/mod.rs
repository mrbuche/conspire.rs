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
        Scalar, Tensor,
        optimize::{EqualityConstraint, OptimizationError, SecondOrderOptimization, SolveStrategy},
    },
};

pub trait HyperelasticIVElements<const G: usize, V, T1, T2, T3, const D: usize>
where
    Self: ElasticIVElements<G, V, T1, T2, T3, D>,
    V: Tensor,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        internal_variables: &InternalVariablesField<G, V>,
    ) -> Result<Scalar, ElementModelError>;
}

impl<B, const G: usize, V, T1, T2, T3, const D: usize> HyperelasticIVElements<G, V, T1, T2, T3, D>
    for Model<B, D>
where
    B: HyperelasticIVElements<G, V, T1, T2, T3, D>,
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
pub trait SecondOrderMinimizeIV<const G: usize, V, T1, T2, T3, F, J, H, X, const D: usize>
where
    V: Tensor,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl SecondOrderOptimization<F, J, H, X>,
        strategy: SolveStrategy,
    ) -> Result<X, OptimizationError>;
}

impl<B, const G: usize, V, T1, T2, T3, const D: usize>
    SecondOrderMinimizeIV<
        G,
        V,
        T1,
        T2,
        T3,
        Scalar,
        NodalForcesSolid<D>,
        NodalStiffnessesSolid<D>,
        NodalCoordinates<D>,
        D,
    > for Model<B, D>
where
    B: HyperelasticIVElements<G, V, T1, T2, T3, D>,
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
        >,
        strategy: SolveStrategy,
    ) -> Result<NodalCoordinates<D>, OptimizationError> {
        let local_solver = match strategy {
            SolveStrategy::Condensed(ref local_solver) => local_solver,
            SolveStrategy::Monolithic { .. } => unimplemented!(
                "The internal variables must be unknowns of the solver to be solved with it."
            ),
        };
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
        let cache: std::cell::RefCell<Option<(NodalCoordinates<D>, InternalVariablesField<G, V>)>> =
            std::cell::RefCell::new(None);
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
            let variables = self.internal_variables_root(local_solver, nodal_coordinates, &warm)?;
            *cache.borrow_mut() = Some((nodal_coordinates.clone(), variables.clone()));
            Ok::<_, ElementModelError>(variables)
        };
        solver.minimize(
            |nodal_coordinates: &NodalCoordinates<D>| {
                Ok(self.helmholtz_free_energy(nodal_coordinates, &solved(nodal_coordinates)?)?)
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
}
