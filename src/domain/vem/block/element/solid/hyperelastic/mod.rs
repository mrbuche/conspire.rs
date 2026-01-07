use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic::Hyperelastic},
    fem::block::element::solid::hyperelastic::HyperelasticFiniteElement,
    math::{Scalar, Tensor},
    vem::block::element::{
        Element, ElementNodalCoordinates, VirtualElement, VirtualElementError,
        solid::{SolidVirtualElement, elastic::ElasticVirtualElement},
    },
};

pub trait HyperelasticVirtualElement<C>
where
    C: Hyperelastic,
    Self: ElasticVirtualElement<C>,
{
    fn helmholtz_free_energy<'a>(
        &'a self,
        constitutive_model: &'a C,
        nodal_coordinates: ElementNodalCoordinates<'a>,
    ) -> Result<Scalar, VirtualElementError>;
}

impl<C> HyperelasticVirtualElement<C> for Element
where
    C: Hyperelastic,
    Self: ElasticVirtualElement<C>,
{
    fn helmholtz_free_energy<'a>(
        &'a self,
        constitutive_model: &'a C,
        nodal_coordinates: ElementNodalCoordinates<'a>,
    ) -> Result<Scalar, VirtualElementError> {
        match self
            .tetrahedra()
            .iter()
            .zip(self.tetrahedra_coordinates(&nodal_coordinates).iter())
            .map(|(tetrahedron, tetrahedron_coordinates)| {
                tetrahedron.helmholtz_free_energy(constitutive_model, tetrahedron_coordinates)
            })
            .sum::<Result<Scalar, _>>()
        {
            Ok(tetrahedra_energy) => {
                match self
                    .deformation_gradients(nodal_coordinates)
                    .iter()
                    .zip(self.integration_weights())
                    .map(|(deformation_gradient, integration_weight)| {
                        Ok::<_, ConstitutiveError>(
                            constitutive_model
                                .helmholtz_free_energy_density(deformation_gradient)?
                                * integration_weight,
                        )
                    })
                    .sum::<Result<Scalar, _>>()
                {
                    Ok(polyhedron_energy) => Ok(polyhedron_energy * (1.0 - self.stabilization())
                        + tetrahedra_energy * self.stabilization()),
                    Err(error) => Err(VirtualElementError::Upstream(
                        format!("{error}"),
                        format!("{self:?}"),
                    )),
                }
            }
            Err(error) => Err(VirtualElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
