#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        Constitutive, ConstitutiveError, Parameters,
        solid::{
            Solid, TWO_THIRDS,
            elastic_viscoplastic::{ElasticViscoplastic, Plastic, Viscoplastic},
            hyperelastic_viscoplastic::HyperelasticViscoplastic,
        },
    },
    math::{IDENTITY, Rank2},
    mechanics::{
        CauchyStress, Deformation, DeformationGradient, DeformationGradientPlastic, Scalar,
    },
};

#[doc = include_str!("doc.md")]
#[derive(Debug)]
pub struct Hencky<P> {
    parameters: P,
}

impl<P> Constitutive<P> for Hencky<P>
where
    P: Parameters,
{
    fn new(parameters: P) -> Self {
        Self { parameters }
    }
}

impl<P> Solid for Hencky<P>
where
    P: Parameters,
{
    fn bulk_modulus(&self) -> &Scalar {
        self.parameters.get(0)
    }
    fn shear_modulus(&self) -> &Scalar {
        self.parameters.get(1)
    }
}

impl<P> Plastic for Hencky<P>
where
    P: Parameters,
{
    fn initial_yield_stress(&self) -> Scalar {
        *self.parameters.get(2)
    }
    fn hardening_slope(&self) -> Scalar {
        *self.parameters.get(3)
    }
}

impl<P> Viscoplastic for Hencky<P>
where
    P: Parameters,
{
    fn rate_sensitivity(&self) -> Scalar {
        *self.parameters.get(4)
    }
    fn reference_flow_rate(&self) -> Scalar {
        *self.parameters.get(5)
    }
}

impl<P> ElasticViscoplastic for Hencky<P>
where
    P: Parameters,
{
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let (deviatoric_strain_e, strain_trace_e) =
            (deformation_gradient_e.left_cauchy_green().logm() * 0.5).deviatoric_and_trace();
        Ok(
            deviatoric_strain_e * (2.0 * self.shear_modulus() / jacobian)
                + IDENTITY * (self.bulk_modulus() * strain_trace_e / jacobian),
        )
    }
}

impl<P> HyperelasticViscoplastic for Hencky<P>
where
    P: Parameters,
{
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<Scalar, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let strain_e = deformation_gradient_e.left_cauchy_green().logm() * 0.5;
        Ok(self.shear_modulus() * strain_e.squared_trace()
            + 0.5
                * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())
                * strain_e.trace().powi(2))
    }
}
