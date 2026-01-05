use crate::{
    constitutive::solid::elastic::Elastic,
    domain::{NodalCoordinates, NodalForcesSolid, NodalStiffnessesSolid},
    fem::{
        Blocks, FiniteElementModel, FiniteElementModelError, FirstOrderRoot, Model,
        block::{
            Block, element::solid::elastic::ElasticFiniteElement,
            solid::elastic::ElasticFiniteElementBlock,
        },
        solid::SolidFiniteElementModel,
    },
    math::optimize::{EqualityConstraint, FirstOrderRootFinding, OptimizationError},
};

pub trait ElasticFiniteElementModel
where
    Self: SolidFiniteElementModel,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<NodalForcesSolid, FiniteElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<NodalStiffnessesSolid, FiniteElementModelError>;
}

impl<B> ElasticFiniteElementModel for Model<B>
where
    B: ElasticFiniteElementModel,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<NodalForcesSolid, FiniteElementModelError> {
        self.blocks.nodal_forces(nodal_coordinates)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<NodalStiffnessesSolid, FiniteElementModelError> {
        self.blocks.nodal_stiffnesses(nodal_coordinates)
    }
}

impl<B1, B2> ElasticFiniteElementModel for Blocks<B1, B2>
where
    B1: ElasticFiniteElementModel,
    B2: ElasticFiniteElementModel,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<NodalForcesSolid, FiniteElementModelError> {
        match Ok::<_, FiniteElementModelError>(
            self.0.nodal_forces(nodal_coordinates)? + self.1.nodal_forces(nodal_coordinates)?,
        ) {
            Ok(nodal_forces) => Ok(nodal_forces),
            Err(error) => Err(FiniteElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<NodalStiffnessesSolid, FiniteElementModelError> {
        match Ok::<_, FiniteElementModelError>(
            self.0.nodal_stiffnesses(nodal_coordinates)?
                + self.1.nodal_stiffnesses(nodal_coordinates)?,
        ) {
            Ok(nodal_stiffnesses) => Ok(nodal_stiffnesses),
            Err(error) => Err(FiniteElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<C, F, const G: usize, const M: usize, const N: usize> ElasticFiniteElementModel
    for Block<C, F, G, M, N>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, M, N>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<NodalForcesSolid, FiniteElementModelError> {
        match ElasticFiniteElementBlock::nodal_forces(self, nodal_coordinates) {
            Ok(nodal_forces) => Ok(nodal_forces),
            Err(error) => Err(FiniteElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<NodalStiffnessesSolid, FiniteElementModelError> {
        match ElasticFiniteElementBlock::nodal_stiffnesses(self, nodal_coordinates) {
            Ok(nodal_stiffnesses) => Ok(nodal_stiffnesses),
            Err(error) => Err(FiniteElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<B> FirstOrderRoot<NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates> for Model<B>
where
    B: ElasticFiniteElementModel,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderRootFinding<NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>,
    ) -> Result<NodalCoordinates, OptimizationError> {
        solver.root(
            |nodal_coordinates: &NodalCoordinates| Ok(self.nodal_forces(nodal_coordinates)?),
            |nodal_coordinates: &NodalCoordinates| Ok(self.nodal_stiffnesses(nodal_coordinates)?),
            self.coordinates().clone().into(),
            equality_constraint,
        )
    }
}
