use crate::{
    fem::{
        Blocks, ElementModel, ElementModelError, Elements, FirstOrderRoot, Model, NodalCoordinates,
        ZerothOrderRoot,
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::{
        Tensor,
        optimize::{
            EqualityConstraint, FirstOrderRootFinding, OptimizationError, ZerothOrderRootFinding,
        },
    },
};

pub trait ElasticElements<const D: usize>
where
    Self: Elements,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError>;
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<NodalForcesSolid<D>, ElementModelError> {
        let mut nodal_forces = NodalForcesSolid::zero(nodal_coordinates.len());
        self.nodal_forces_into(nodal_coordinates, &mut nodal_forces)?;
        Ok(nodal_forces)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError>;
}

impl<B, const D: usize> ElasticElements<D> for Model<B, D>
where
    B: ElasticElements<D>,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.blocks.nodal_forces_into(nodal_coordinates, nodal_forces)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError> {
        self.blocks.nodal_stiffnesses(nodal_coordinates)
    }
}

impl<B1, B2, const D: usize> ElasticElements<D> for Blocks<B1, B2>
where
    B1: ElasticElements<D>,
    B2: ElasticElements<D>,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.0.nodal_forces_into(nodal_coordinates, nodal_forces)?;
        self.1.nodal_forces_into(nodal_coordinates, nodal_forces)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError> {
        Ok(self.0.nodal_stiffnesses(nodal_coordinates)?
            + self.1.nodal_stiffnesses(nodal_coordinates)?)
    }
}

impl<B, const D: usize> ZerothOrderRoot<NodalCoordinates<D>> for Model<B, D>
where
    B: ElasticElements<D>,
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
    B: ElasticElements<D>,
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
