use crate::{
    constitutive::solid::elastic_hyperviscous::ElasticHyperviscous,
    fem::{
        FiniteElementModelError, NodalCoordinates, NodalVelocities,
        block::{Block, element::solid::elastic_hyperviscous::ElasticHyperviscousFiniteElement},
        solid::{
            elastic_hyperviscous::ElasticHyperviscousFiniteElements,
            viscoelastic::ViscoelasticFiniteElements,
        },
    },
    math::Scalar,
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    ElasticHyperviscousFiniteElements<3> for Block<C, F, G, M, N, P>
where
    C: ElasticHyperviscous,
    F: ElasticHyperviscousFiniteElement<C, G, M, N, P>,
    Self: ViscoelasticFiniteElements<3>,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Result<Scalar, FiniteElementModelError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, nodes)| {
                element.viscous_dissipation(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    &Self::element_coordinates(nodal_velocities, nodes),
                )
            })
            .sum()
        {
            Ok(viscous_dissipation) => Ok(viscous_dissipation),
            Err(error) => Err(FiniteElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Result<Scalar, FiniteElementModelError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, nodes)| {
                element.dissipation_potential(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    &Self::element_coordinates(nodal_velocities, nodes),
                )
            })
            .sum()
        {
            Ok(dissipation_potential) => Ok(dissipation_potential),
            Err(error) => Err(FiniteElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
