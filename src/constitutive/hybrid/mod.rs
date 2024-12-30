//! Hybrid constitutive models.

mod elastic;
mod hyperelastic;

use crate::{
    constitutive::{Constitutive, ConstitutiveError},
    mechanics::DeformationGradient,
};

/// Required methods for hybrid constitutive models.
pub trait Hybrid<'a, C1: Constitutive<'a>, C2: Constitutive<'a>> {
    /// Constructs and returns a new hybrid constitutive model.
    fn construct(constitutive_model_1: C1, constitutive_model_2: C2) -> Self;
    /// Returns a reference to the first constitutive model.
    fn get_constitutive_model_1(&self) -> &C1;
    /// Returns a reference to the second constitutive model.
    fn get_constitutive_model_2(&self) -> &C2;
}

/// A hybrid constitutive model based on the additive decomposition.
#[derive(Debug)]
pub struct Additive<C1, C2> {
    constitutive_model_1: C1,
    constitutive_model_2: C2,
}

/// A hybrid constitutive model based on the multiplicative decomposition.
#[derive(Debug)]
pub struct Multiplicative<C1, C2> {
    constitutive_model_1: C1,
    constitutive_model_2: C2,
}

/// Required methods for hybrid constitutive models based on the multiplicative decomposition.
pub trait MultiplicativeTrait {
    fn calculate_deformation_gradients(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<(DeformationGradient, DeformationGradient), ConstitutiveError>;
}

impl<'a, C1: Constitutive<'a>, C2: Constitutive<'a>> Hybrid<'a, C1, C2> for Additive<C1, C2> {
    fn construct(constitutive_model_1: C1, constitutive_model_2: C2) -> Self {
        Self {
            constitutive_model_1,
            constitutive_model_2,
        }
    }
    fn get_constitutive_model_1(&self) -> &C1 {
        &self.constitutive_model_1
    }
    fn get_constitutive_model_2(&self) -> &C2 {
        &self.constitutive_model_2
    }
}

impl<'a, C1: Constitutive<'a>, C2: Constitutive<'a>> Hybrid<'a, C1, C2> for Multiplicative<C1, C2> {
    fn construct(constitutive_model_1: C1, constitutive_model_2: C2) -> Self {
        Self {
            constitutive_model_1,
            constitutive_model_2,
        }
    }
    fn get_constitutive_model_1(&self) -> &C1 {
        &self.constitutive_model_1
    }
    fn get_constitutive_model_2(&self) -> &C2 {
        &self.constitutive_model_2
    }
}
