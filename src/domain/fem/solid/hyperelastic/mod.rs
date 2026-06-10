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

pub trait HyperelasticFiniteElements
where
    Self: ElasticFiniteElements,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Scalar, FiniteElementModelError>;
}

impl<B> HyperelasticFiniteElements for Model<B>
where
    B: HyperelasticFiniteElements,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Scalar, FiniteElementModelError> {
        self.blocks.helmholtz_free_energy(nodal_coordinates)
    }
}

impl<B1, B2> HyperelasticFiniteElements for Blocks<B1, B2>
where
    B1: HyperelasticFiniteElements,
    B2: HyperelasticFiniteElements,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Scalar, FiniteElementModelError> {
        Ok(self.0.helmholtz_free_energy(nodal_coordinates)?
            + self.1.helmholtz_free_energy(nodal_coordinates)?)
    }
}

impl<B> FirstOrderMinimize<Scalar, NodalCoordinates> for Model<B>
where
    B: HyperelasticFiniteElements,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderOptimization<Scalar, NodalCoordinates>,
    ) -> Result<NodalCoordinates, OptimizationError> {
        solver.minimize(
            |nodal_coordinates: &NodalCoordinates| {
                Ok(self.helmholtz_free_energy(nodal_coordinates)?)
            },
            |nodal_coordinates: &NodalCoordinates| Ok(self.nodal_forces(nodal_coordinates)?),
            self.coordinates().clone().into(),
            equality_constraint,
        )
    }
}

impl<B> SecondOrderMinimize<Scalar, NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>
    for Model<B>
where
    B: HyperelasticFiniteElements,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            NodalCoordinates,
        >,
    ) -> Result<NodalCoordinates, OptimizationError> {
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let banded = band_from_neighbors(&neighbors, &equality_constraint, 3);
        solver.minimize(
            |nodal_coordinates: &NodalCoordinates| {
                Ok(self.helmholtz_free_energy(nodal_coordinates)?)
            },
            |nodal_coordinates: &NodalCoordinates| Ok(self.nodal_forces(nodal_coordinates)?),
            |nodal_coordinates: &NodalCoordinates| Ok(self.nodal_stiffnesses(nodal_coordinates)?),
            self.coordinates().clone().into(),
            equality_constraint,
            Some(banded),
        )
    }
}
