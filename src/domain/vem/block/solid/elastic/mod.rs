use crate::{
    constitutive::solid::elastic::Elastic,
    math::{
        Tensor,
        optimize::{
            EqualityConstraint, FirstOrderRootFinding, OptimizationError, ZerothOrderRootFinding,
        },
    },
    vem::{
        NodalCoordinates,
        block::{
            Block, FirstOrderRoot, VirtualElementBlockError, ZerothOrderRoot,
            element::{VirtualElementError, solid::elastic::ElasticVirtualElement},
            solid::{NodalForcesSolid, NodalStiffnessesSolid, SolidVirtualElementBlock},
        },
    },
};

pub trait ElasticVirtualElementBlock<C, F>
where
    C: Elastic,
    F: ElasticVirtualElement<C>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<NodalForcesSolid, VirtualElementBlockError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<NodalStiffnessesSolid, VirtualElementBlockError>;
}

impl<C, F> ElasticVirtualElementBlock<C, F> for Block<C, F>
where
    C: Elastic,
    F: ElasticVirtualElement<C>,
    Self: SolidVirtualElementBlock<C, F>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<NodalForcesSolid, VirtualElementBlockError> {
        let mut nodal_forces = NodalForcesSolid::zero(nodal_coordinates.len());
        match self
            .elements()
            .iter()
            .zip(self.element_faces())
            .try_for_each(|(element, faces)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        self.element_coordinates(nodal_coordinates, faces),
                    )?
                    .iter()
                    .zip(self.element_nodes(faces))
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), VirtualElementError>(())
            }) {
            Ok(()) => Ok(nodal_forces),
            Err(error) => Err(VirtualElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<NodalStiffnessesSolid, VirtualElementBlockError> {
        let mut nodal_stiffnesses = NodalStiffnessesSolid::zero(nodal_coordinates.len());
        match self
            .elements()
            .iter()
            .zip(self.element_faces())
            .try_for_each(|(element, faces)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        self.element_coordinates(nodal_coordinates, faces),
                    )?
                    .iter()
                    .zip(self.element_nodes(faces))
                    .for_each(|(object, &node_a)| {
                        object.iter().zip(self.element_nodes(faces)).for_each(
                            |(nodal_stiffness, &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                            },
                        )
                    });
                Ok::<(), VirtualElementError>(())
            }) {
            Ok(()) => Ok(nodal_stiffnesses),
            Err(error) => Err(VirtualElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<C, F> ZerothOrderRoot<C, F, NodalCoordinates> for Block<C, F>
where
    C: Elastic,
    F: ElasticVirtualElement<C>,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl ZerothOrderRootFinding<NodalCoordinates>,
    ) -> Result<NodalCoordinates, OptimizationError> {
        solver.root(
            |nodal_coordinates: &NodalCoordinates| Ok(self.nodal_forces(nodal_coordinates)?),
            self.coordinates().clone().into(),
            equality_constraint,
        )
    }
}

impl<C, F> FirstOrderRoot<C, F, NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>
    for Block<C, F>
where
    C: Elastic,
    F: ElasticVirtualElement<C>,
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
