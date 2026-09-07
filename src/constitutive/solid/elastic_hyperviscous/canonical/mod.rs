#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        canonical::Canonical,
        fluid::hyperviscous::Hyperviscous,
        solid::{elastic::Elastic, elastic_hyperviscous::ElasticHyperviscous},
    },
    math::Quantity,
    mechanics::{DeformationGradient, DeformationGradientRate},
    units::Dissipation,
};

impl<C1, C2> ElasticHyperviscous for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Hyperviscous,
{
    fn viscous_dissipation(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        self.1
            .viscous_dissipation(deformation_gradient, deformation_gradient_rate)
    }
}
