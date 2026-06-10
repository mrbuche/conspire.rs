use crate::{
    constitutive::solid::viscoelastic::Viscoelastic,
    fem::{
        FiniteElementModelError, NodalCoordinates, NodalVelocities,
        block::{
            Block,
            element::{FiniteElementError, solid::viscoelastic::ViscoelasticFiniteElement},
        },
        solid::{
            NodalForcesSolid, NodalStiffnessesSolid, SolidFiniteElements,
            viscoelastic::ViscoelasticFiniteElements,
        },
    },
    math::Tensor,
    mechanics::DeformationGradientRateList,
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> Block<C, F, G, M, N, P>
where
    C: Viscoelastic,
    F: ViscoelasticFiniteElement<C, G, M, N, P>,
{
    pub fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Vec<DeformationGradientRateList<G>> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, nodes)| {
                element.deformation_gradient_rates(
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    &Self::element_coordinates(nodal_velocities, nodes),
                )
            })
            .collect()
    }
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    ViscoelasticFiniteElements for Block<C, F, G, M, N, P>
where
    C: Viscoelastic,
    F: ViscoelasticFiniteElement<C, G, M, N, P>,
    Self: SolidFiniteElements,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<NodalForcesSolid, FiniteElementModelError> {
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
                        &Self::element_coordinates(nodal_velocities, nodes),
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
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<NodalStiffnessesSolid, FiniteElementModelError> {
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
                        &Self::element_coordinates(nodal_velocities, nodes),
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
