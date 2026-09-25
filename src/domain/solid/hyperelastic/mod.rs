use crate::{
    domain::{
        Blocks, ElementModel, ElementModelError, FirstOrderMinimize, Model, NodalCoordinates,
        ProvidesTangent, SecondOrderMinimize, SolverFor,
        block::{
            element::Elements,
            feti::element_systems::{DecomposableElements, ElementSystems},
            finalize_node_neighbors, solver_from_neighbors,
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolidSymmetric, elastic::ElasticElements},
    },
    math::{
        Quantity, Tensor,
        optimize::{
            EqualityConstraint, FirstOrderOptimization, NewtonRaphson, OptimizationError,
            SecondOrderOptimization,
        },
    },
    units::Energy,
};

pub trait HyperelasticElements<const D: usize>
where
    Self: ElasticElements<D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Quantity<Energy>, ElementModelError>;
    fn nodal_stiffnesses_symmetric_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_stiffnesses: &mut NodalStiffnessesSolidSymmetric<D>,
    ) -> Result<(), ElementModelError>;
    fn nodal_stiffnesses_symmetric(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<NodalStiffnessesSolidSymmetric<D>, ElementModelError> {
        let mut nodal_stiffnesses = NodalStiffnessesSolidSymmetric::zero(nodal_coordinates.len());
        self.nodal_stiffnesses_symmetric_into(nodal_coordinates, &mut nodal_stiffnesses)?;
        Ok(nodal_stiffnesses)
    }
}

impl<B, const D: usize> HyperelasticElements<D> for Model<B, D>
where
    B: HyperelasticElements<D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.blocks.helmholtz_free_energy(nodal_coordinates)
    }
    fn nodal_stiffnesses_symmetric_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_stiffnesses: &mut NodalStiffnessesSolidSymmetric<D>,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_stiffnesses_symmetric_into(nodal_coordinates, nodal_stiffnesses)
    }
}

impl<B1, B2, const D: usize> HyperelasticElements<D> for Blocks<B1, B2>
where
    B1: HyperelasticElements<D>,
    B2: HyperelasticElements<D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        Ok(self.0.helmholtz_free_energy(nodal_coordinates)?
            + self.1.helmholtz_free_energy(nodal_coordinates)?)
    }
    fn nodal_stiffnesses_symmetric_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_stiffnesses: &mut NodalStiffnessesSolidSymmetric<D>,
    ) -> Result<(), ElementModelError> {
        self.0
            .nodal_stiffnesses_symmetric_into(nodal_coordinates, nodal_stiffnesses)?;
        self.1
            .nodal_stiffnesses_symmetric_into(nodal_coordinates, nodal_stiffnesses)
    }
}

impl<B, const D: usize>
    FirstOrderMinimize<Quantity<Energy>, NodalForcesSolid<D>, NodalCoordinates<D>> for Model<B, D>
where
    B: HyperelasticElements<D>,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderOptimization<Quantity<Energy>, NodalForcesSolid<D>, NodalCoordinates<D>>,
    ) -> Result<NodalCoordinates<D>, OptimizationError> {
        solver.minimize(
            |nodal_coordinates: &NodalCoordinates<D>| {
                Ok(self.helmholtz_free_energy(nodal_coordinates)?)
            },
            |nodal_coordinates: &NodalCoordinates<D>| Ok(self.nodal_forces(nodal_coordinates)?),
            self.coordinates().clone().into(),
            equality_constraint,
        )
    }
}

impl<B, const D: usize> SolverFor<Model<B, D>, Quantity<Energy>, NodalForcesSolid<D>>
    for NewtonRaphson
where
    B: HyperelasticElements<D>,
{
    type Tangent = NodalStiffnessesSolidSymmetric<D>;
    const SPARSE: bool = true;
}

impl<B, const D: usize> ProvidesTangent<NodalCoordinates<D>, NodalStiffnessesSolidSymmetric<D>>
    for Model<B, D>
where
    B: HyperelasticElements<D>,
{
    fn provide_tangent(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<NodalStiffnessesSolidSymmetric<D>, ElementModelError> {
        self.nodal_stiffnesses_symmetric(nodal_coordinates)
    }
}

impl<B> ProvidesTangent<NodalCoordinates<3>, ElementSystems> for Model<B, 3>
where
    B: DecomposableElements,
{
    fn provide_tangent(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<ElementSystems, ElementModelError> {
        self.element_systems(nodal_coordinates)
    }
}

impl<B, const D: usize>
    SecondOrderMinimize<Quantity<Energy>, NodalForcesSolid<D>, NodalCoordinates<D>> for Model<B, D>
where
    B: HyperelasticElements<D>,
{
    fn minimize<S>(
        &self,
        equality_constraint: EqualityConstraint,
        solver: S,
    ) -> Result<NodalCoordinates<D>, OptimizationError>
    where
        S: SolverFor<Self, Quantity<Energy>, NodalForcesSolid<D>>
            + SecondOrderOptimization<
                Quantity<Energy>,
                NodalForcesSolid<D>,
                S::Tangent,
                NodalCoordinates<D>,
            >,
        Self: ProvidesTangent<NodalCoordinates<D>, S::Tangent>,
    {
        let sparse = S::SPARSE.then(|| {
            let mut neighbors = vec![Vec::new(); self.coordinates().len()];
            self.node_neighbors(&mut neighbors);
            finalize_node_neighbors(&mut neighbors);
            solver_from_neighbors(&neighbors, &equality_constraint, D, true)
        });
        solver.minimize(
            |nodal_coordinates: &NodalCoordinates<D>| {
                Ok(self.helmholtz_free_energy(nodal_coordinates)?)
            },
            |nodal_coordinates: &NodalCoordinates<D>| Ok(self.nodal_forces(nodal_coordinates)?),
            |nodal_coordinates: &NodalCoordinates<D>| Ok(self.provide_tangent(nodal_coordinates)?),
            self.coordinates().clone().into(),
            equality_constraint,
            sparse,
        )
    }
}
