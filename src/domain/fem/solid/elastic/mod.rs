use crate::{
    fem::{
        Blocks, FiniteElementModel, FiniteElementModelError, FirstOrderRoot, Model,
        NodalCoordinates, ZerothOrderRoot,
        solid::{NodalForcesSolid, NodalStiffnessesSolid, SolidFiniteElements},
    },
    math::optimize::{
        EqualityConstraint, FirstOrderRootFinding, OptimizationError, ZerothOrderRootFinding,
    },
};

pub trait ElasticFiniteElements<const D: usize>
where
    Self: SolidFiniteElements,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<NodalForcesSolid<D>, FiniteElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<NodalStiffnessesSolid<D>, FiniteElementModelError>;
}

impl<B, const D: usize> ElasticFiniteElements<D> for Model<B, D>
where
    B: ElasticFiniteElements<D>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<NodalForcesSolid<D>, FiniteElementModelError> {
        self.blocks.nodal_forces(nodal_coordinates)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<NodalStiffnessesSolid<D>, FiniteElementModelError> {
        self.blocks.nodal_stiffnesses(nodal_coordinates)
    }
}

impl<B1, B2, const D: usize> ElasticFiniteElements<D> for Blocks<B1, B2>
where
    B1: ElasticFiniteElements<D>,
    B2: ElasticFiniteElements<D>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<NodalForcesSolid<D>, FiniteElementModelError> {
        Ok(self.0.nodal_forces(nodal_coordinates)? + self.1.nodal_forces(nodal_coordinates)?)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<NodalStiffnessesSolid<D>, FiniteElementModelError> {
        Ok(self.0.nodal_stiffnesses(nodal_coordinates)?
            + self.1.nodal_stiffnesses(nodal_coordinates)?)
    }
}

impl<B, const D: usize> ZerothOrderRoot<NodalCoordinates<D>> for Model<B, D>
where
    B: ElasticFiniteElements<D>,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl ZerothOrderRootFinding<NodalCoordinates<D>>,
    ) -> Result<NodalCoordinates<D>, OptimizationError> {
        solver.root(
            |nodal_coordinates: &NodalCoordinates<D>| Ok(self.nodal_forces(nodal_coordinates)?),
            self.coordinates().clone().into(),
            equality_constraint,
        )
    }
}

impl<B, const D: usize>
    FirstOrderRoot<NodalForcesSolid<D>, NodalStiffnessesSolid<D>, NodalCoordinates<D>>
    for Model<B, D>
where
    B: ElasticFiniteElements<D>,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
    ) -> Result<NodalCoordinates<D>, OptimizationError> {
        solver.root(
            |nodal_coordinates: &NodalCoordinates<D>| Ok(self.nodal_forces(nodal_coordinates)?),
            |nodal_coordinates: &NodalCoordinates<D>| {
                Ok(self.nodal_stiffnesses(nodal_coordinates)?)
            },
            self.coordinates().clone().into(),
            equality_constraint,
        )
    }
}
