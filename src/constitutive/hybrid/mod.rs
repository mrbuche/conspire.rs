//! Hybrid constitutive models.

mod elastic;
mod elastic_viscoplastic;
mod hyperelastic;
mod hyperelastic_viscoplastic;

use crate::{constitutive::ConstitutiveError, mechanics::DeformationGradient};

/// A hybrid constitutive model based on the additive decomposition.
#[derive(Debug)]
pub struct Additive<C1, C2>(C1, C2);

impl<C1, C2> From<(C1, C2)> for Additive<C1, C2> {
    fn from((constitutive_model_1, constitutive_model_2): (C1, C2)) -> Self {
        Self(constitutive_model_1, constitutive_model_2)
    }
}

/// A hybrid constitutive model based on the multiplicative decomposition.
#[derive(Debug)]
pub struct Multiplicative<C1, C2>(C1, C2);

impl<C1, C2> From<(C1, C2)> for Multiplicative<C1, C2> {
    fn from((constitutive_model_1, constitutive_model_2): (C1, C2)) -> Self {
        Self(constitutive_model_1, constitutive_model_2)
    }
}

/// Required methods for hybrid constitutive models based on the multiplicative decomposition.
pub trait MultiplicativeTrait {
    fn deformation_gradients(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<(DeformationGradient, DeformationGradient), ConstitutiveError>;
}
