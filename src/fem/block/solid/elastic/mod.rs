use crate::{
    constitutive::solid::elastic::Elastic,
    fem::{
        NodalCoordinatesBlock, NodalForcesSolid, NodalStiffnessesSolid,
        block::{
            ElementBlock, FiniteElementBlockError, FirstOrderRoot, ZerothOrderRoot,
            element::{ElasticFiniteElement, FiniteElementError},
            solid::SolidFiniteElementBlock,
        },
    },
    math::{
        Tensor,
        optimize::{
            EqualityConstraint, FirstOrderRootFinding, OptimizationError, ZerothOrderRootFinding,
        },
    },
};

pub trait ElasticFiniteElementBlock<C, F, const G: usize, const N: usize>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, N>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<NodalForcesSolid, FiniteElementBlockError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<NodalStiffnessesSolid, FiniteElementBlockError>;
}

impl<C, F, const G: usize, const N: usize> ElasticFiniteElementBlock<C, F, G, N>
    for ElementBlock<C, F, N>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, N>,
    Self: SolidFiniteElementBlock<C, F, G, N>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<NodalForcesSolid, FiniteElementBlockError> {
        let mut nodal_forces = NodalForcesSolid::zero(nodal_coordinates.len());
        match self
            .elements()
            .iter()
            .zip(self.connectivity().iter())
            .try_for_each(|(element, element_connectivity)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                    )?
                    .iter()
                    .zip(element_connectivity.iter())
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), FiniteElementError>(())
            }) {
            Ok(()) => Ok(nodal_forces),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<NodalStiffnessesSolid, FiniteElementBlockError> {
        let mut nodal_stiffnesses = NodalStiffnessesSolid::zero(nodal_coordinates.len());
        match self
            .elements()
            .iter()
            .zip(self.connectivity().iter())
            .try_for_each(|(element, element_connectivity)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                    )?
                    .iter()
                    .zip(element_connectivity.iter())
                    .for_each(|(object, &node_a)| {
                        object.iter().zip(element_connectivity.iter()).for_each(
                            |(nodal_stiffness, &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                            },
                        )
                    });
                Ok::<(), FiniteElementError>(())
            }) {
            Ok(()) => Ok(nodal_stiffnesses),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<C, F, const G: usize, const N: usize> ZerothOrderRoot<C, F, G, N, NodalCoordinatesBlock>
    for ElementBlock<C, F, N>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, N>,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl ZerothOrderRootFinding<NodalCoordinatesBlock>,
    ) -> Result<NodalCoordinatesBlock, OptimizationError> {
        solver.root(
            |nodal_coordinates: &NodalCoordinatesBlock| Ok(self.nodal_forces(nodal_coordinates)?),
            self.coordinates().clone().into(),
            equality_constraint,
        )
    }
}

impl<C, F, const G: usize, const N: usize>
    FirstOrderRoot<C, F, G, N, NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinatesBlock>
    for ElementBlock<C, F, N>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, N>,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid,
            NodalStiffnessesSolid,
            NodalCoordinatesBlock,
        >,
    ) -> Result<NodalCoordinatesBlock, OptimizationError> {
        solver.root(
            |nodal_coordinates: &NodalCoordinatesBlock| Ok(self.nodal_forces(nodal_coordinates)?),
            |nodal_coordinates: &NodalCoordinatesBlock| {
                Ok(self.nodal_stiffnesses(nodal_coordinates)?)
            },
            self.coordinates().clone().into(),
            equality_constraint,
        )
    }
}
