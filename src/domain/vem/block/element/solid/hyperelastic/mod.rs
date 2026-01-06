use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic::Hyperelastic},
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
        //
        // NEED BETA
        let beta = 0.1;
        //
        // GET COORDINATES FOR TETRAHEDRA HERE
        //
        // GET RID OF UNWRAPS
        //
        match Ok::<_, ConstitutiveError>(
            self.deformation_gradients(nodal_coordinates)
                .iter()
                .zip(self.integration_weights())
                .map(|(deformation_gradient, integration_weight)| {
                    Ok::<_, ConstitutiveError>(
                        constitutive_model.helmholtz_free_energy_density(deformation_gradient)?
                            * integration_weight,
                    )
                })
                .sum::<Result<Scalar, _>>()
                .unwrap()
                * (1.0 - beta)
                + self
                    .tetrahedra_deformation_gradients_and_volumes(todo!())
                    .iter()
                    .map(|(deformation_gradient, integration_weight)| {
                        Ok::<_, ConstitutiveError>(
                            constitutive_model
                                .helmholtz_free_energy_density(deformation_gradient)?
                                * integration_weight,
                        )
                    })
                    .sum::<Result<Scalar, _>>()
                    .unwrap()
                    * beta,
        ) {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(VirtualElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
