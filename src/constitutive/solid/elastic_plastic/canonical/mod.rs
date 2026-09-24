#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        canonical::Canonical,
        fluid::plastic::{PlasticStateVariables, RateIndependentPlastic, YieldSurface},
        solid::{elastic::Elastic, elastic_plastic::ElasticPlastic},
    },
    math::Quantity,
    mechanics::{FlowDirectionPlastic, MandelStressElastic, StretchingRatePlastic},
    units::{Dissipation, Rate, Stress},
};

impl<C1, C2> YieldSurface for Canonical<C1, C2>
where
    C1: Elastic,
    C2: YieldSurface,
{
    fn equivalent_stress(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        self.1.equivalent_stress(deviatoric_mandel_stress)
    }
    fn flow_direction(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError> {
        self.1.flow_direction(deviatoric_mandel_stress)
    }
    fn flow_direction_slope(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
        increment: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError> {
        self.1
            .flow_direction_slope(deviatoric_mandel_stress, increment)
    }
    fn dissipation_potential(
        &self,
        plastic_stretching_rate: StretchingRatePlastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        self.1
            .dissipation_potential(plastic_stretching_rate, yield_stress)
    }
}

impl<C1, C2> RateIndependentPlastic for Canonical<C1, C2>
where
    C1: Elastic,
    C2: RateIndependentPlastic,
{
    fn initial_state(&self) -> PlasticStateVariables {
        self.1.initial_state()
    }
    fn yield_function(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        self.1
            .yield_function(deviatoric_mandel_stress, equivalent_plastic_strain)
    }
    fn plastic_stretching_rate(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
        plastic_multiplier: Quantity<Rate>,
    ) -> Result<StretchingRatePlastic, ConstitutiveError> {
        self.1
            .plastic_stretching_rate(deviatoric_mandel_stress, plastic_multiplier)
    }
}

impl<C1, C2> ElasticPlastic for Canonical<C1, C2>
where
    C1: Elastic,
    C2: RateIndependentPlastic,
{
}
