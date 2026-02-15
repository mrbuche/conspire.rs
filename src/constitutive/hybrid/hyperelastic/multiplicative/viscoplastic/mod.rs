#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::viscoplastic::Viscoplastic,
        hybrid::ElasticMultiplicativeViscoplastic,
        solid::{hyperelastic::Hyperelastic, hyperelastic_viscoplastic::HyperelasticViscoplastic},
    },
    math::Tensor,
    mechanics::{DeformationGradient, DeformationGradientPlastic, Scalar},
};

impl<C1, C2, Y2> HyperelasticViscoplastic<Y2> for ElasticMultiplicativeViscoplastic<C1, C2, Y2>
where
    C1: Hyperelastic,
    C2: Viscoplastic<Y2>,
    Y2: Tensor,
{
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<Scalar, ConstitutiveError> {
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        self.0
            .helmholtz_free_energy_density(&deformation_gradient_e.into())
    }
}
