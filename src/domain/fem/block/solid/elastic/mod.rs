use crate::{
    constitutive::solid::elastic::Elastic,
    fem::{
        FiniteElementModelError, NodalCoordinates,
        block::{
            Block,
            element::{FiniteElementError, solid::elastic::ElasticFiniteElement},
        },
        solid::{
            NodalForcesSolid, NodalStiffnessesSolid, SolidFiniteElements,
            elastic::ElasticFiniteElements,
        },
    },
    math::Tensor,
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> ElasticFiniteElements<3>
    for Block<C, F, G, M, N, P>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, M, N, P>,
    Self: SolidFiniteElements,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<NodalForcesSolid<3>, FiniteElementModelError> {
        let mut nodal_forces = NodalForcesSolid::zero(nodal_coordinates.len());
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), FiniteElementError>(())
            }) {
            Ok(()) => Ok(nodal_forces),
            Err(error) => Err(FiniteElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<NodalStiffnessesSolid<3>, FiniteElementModelError> {
        let mut nodal_stiffnesses = NodalStiffnessesSolid::zero(nodal_coordinates.len());
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(object, &node_a)| {
                        object
                            .into_iter()
                            .zip(nodes)
                            .for_each(|(nodal_stiffness, &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                            })
                    });
                Ok::<(), FiniteElementError>(())
            }) {
            Ok(()) => Ok(nodal_stiffnesses),
            Err(error) => Err(FiniteElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
