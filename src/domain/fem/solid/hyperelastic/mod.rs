use crate::{
    fem::{
        Blocks, FiniteElementModel, FiniteElementModelError, FiniteElements, FirstOrderMinimize,
        Model, NodalCoordinates, SecondOrderMinimize,
        block::{band_from_neighbors, finalize_node_neighbors},
        solid::{NodalForcesSolid, NodalStiffnessesSolid, elastic::ElasticFiniteElements},
    },
    math::{
        Scalar, Tensor,
        optimize::{
            EqualityConstraint, FirstOrderOptimization, OptimizationError, SecondOrderOptimization,
        },
    },
};

pub trait HyperelasticFiniteElements<const D: usize>
where
    Self: ElasticFiniteElements<D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Scalar, FiniteElementModelError>;
}

impl<B, const D: usize> HyperelasticFiniteElements<D> for Model<B, D>
where
    B: HyperelasticFiniteElements<D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Scalar, FiniteElementModelError> {
        self.blocks.helmholtz_free_energy(nodal_coordinates)
    }
}

impl<B1, B2, const D: usize> HyperelasticFiniteElements<D> for Blocks<B1, B2>
where
    B1: HyperelasticFiniteElements<D>,
    B2: HyperelasticFiniteElements<D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Scalar, FiniteElementModelError> {
        Ok(self.0.helmholtz_free_energy(nodal_coordinates)?
            + self.1.helmholtz_free_energy(nodal_coordinates)?)
    }
}

impl<B, const D: usize> FirstOrderMinimize<Scalar, NodalCoordinates<D>> for Model<B, D>
where
    B: HyperelasticFiniteElements<D>,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderOptimization<Scalar, NodalCoordinates<D>>,
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

impl<B, const D: usize>
    SecondOrderMinimize<Scalar, NodalForcesSolid<D>, NodalStiffnessesSolid<D>, NodalCoordinates<D>>
    for Model<B, D>
where
    B: HyperelasticFiniteElements<D>,
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
    ) -> Result<NodalCoordinates<D>, OptimizationError> {
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let banded = band_from_neighbors(&neighbors, &equality_constraint, D);
        solver.minimize(
            |nodal_coordinates: &NodalCoordinates<D>| {
                Ok(self.helmholtz_free_energy(nodal_coordinates)?)
            },
            |nodal_coordinates: &NodalCoordinates<D>| Ok(self.nodal_forces(nodal_coordinates)?),
            |nodal_coordinates: &NodalCoordinates<D>| {
                Ok(self.nodal_stiffnesses(nodal_coordinates)?)
            },
            self.coordinates().clone().into(),
            equality_constraint,
            Some(banded),
        )
    }
}
