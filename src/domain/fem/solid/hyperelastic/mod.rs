pub mod internal_variables;

use crate::math::{EnergyDensity, Quantity};
use crate::{
    fem::{
        Blocks, ElementModel, ElementModelError, Elements, FirstOrderMinimize, Model,
        NodalCoordinates, SecondOrderMinimize,
        block::{finalize_node_neighbors, solver_from_neighbors},
        solid::{NodalForcesSolid, NodalStiffnessesSolidSymmetric, elastic::ElasticElements},
    },
    math::{
        Scalar, Tensor,
        optimize::{
            EqualityConstraint, FirstOrderOptimization, OptimizationError, SecondOrderOptimization,
        },
    },
};

pub trait HyperelasticElements<const D: usize>
where
    Self: ElasticElements<D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Quantity<EnergyDensity>, ElementModelError>;
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
    ) -> Result<Quantity<EnergyDensity>, ElementModelError> {
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
    ) -> Result<Quantity<EnergyDensity>, ElementModelError> {
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

impl<B, const D: usize> FirstOrderMinimize<Scalar, NodalForcesSolid<D>, NodalCoordinates<D>>
    for Model<B, D>
where
    B: HyperelasticElements<D>,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderOptimization<Scalar, NodalForcesSolid<D>, NodalCoordinates<D>>,
    ) -> Result<NodalCoordinates<D>, OptimizationError> {
        solver.minimize(
            |nodal_coordinates: &NodalCoordinates<D>| {
                // The solver only compares its objective, so the free energy is
                // spent where it is handed over.
                Ok(self
                    .helmholtz_free_energy(nodal_coordinates)?
                    .value_as::<EnergyDensity>())
            },
            |nodal_coordinates: &NodalCoordinates<D>| Ok(self.nodal_forces(nodal_coordinates)?),
            self.coordinates().clone().into(),
            equality_constraint,
        )
    }
}

impl<B, const D: usize>
    SecondOrderMinimize<
        Scalar,
        NodalForcesSolid<D>,
        NodalStiffnessesSolidSymmetric<D>,
        NodalCoordinates<D>,
    > for Model<B, D>
where
    B: HyperelasticElements<D>,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid<D>,
            NodalStiffnessesSolidSymmetric<D>,
            NodalCoordinates<D>,
        >,
    ) -> Result<NodalCoordinates<D>, OptimizationError> {
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let sparse = solver_from_neighbors(&neighbors, &equality_constraint, D, true);
        solver.minimize(
            |nodal_coordinates: &NodalCoordinates<D>| {
                // The solver only compares its objective, so the free energy is
                // spent where it is handed over.
                Ok(self
                    .helmholtz_free_energy(nodal_coordinates)?
                    .value_as::<EnergyDensity>())
            },
            |nodal_coordinates: &NodalCoordinates<D>| Ok(self.nodal_forces(nodal_coordinates)?),
            |nodal_coordinates: &NodalCoordinates<D>| {
                Ok(self.nodal_stiffnesses_symmetric(nodal_coordinates)?)
            },
            self.coordinates().clone().into(),
            equality_constraint,
            Some(sparse),
        )
    }
}
