#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        canonical::Canonical,
        fluid::{
            plastic::Plastic,
            viscoplastic::{
                Viscoplastic, ViscoplasticAlgebraRate, ViscoplasticEvolution,
                ViscoplasticStateVariables,
            },
        },
        solid::{
            elastic::Elastic,
            elastic_viscoplastic::{ElasticPlasticOrViscoplastic, ElasticViscoplastic},
        },
    },
    math::{
        ContractFirstSecondWithSecond, ContractSecondWithFirst, Differentiate, Intermediate,
        Quantity, Rank2, Reference, Scalar, Tensor, TensorTuple,
        integrate::{Flat, Product, StateEvolution, Unimodular},
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, CauchyTangentStiffnessElastic, DeformationGradient,
        DeformationGradientPlastic, FirstPiolaKirchhoffStress, FirstPiolaKirchhoffStressElastic,
        FirstPiolaKirchhoffTangentStiffness, FirstPiolaKirchhoffTangentStiffnessElastic,
        MandelStressElastic, SecondPiolaKirchhoffStress, SecondPiolaKirchhoffStressElastic,
        SecondPiolaKirchhoffTangentStiffness, SecondPiolaKirchhoffTangentStiffnessElastic,
        StretchingRatePlastic,
    },
    units::{Dissipation, Rate, Stress, Time},
};

impl<C1, C2> Plastic for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Plastic,
{
    fn initial_yield_stress(&self) -> Quantity<Stress> {
        self.1.initial_yield_stress()
    }
    fn hardening_slope(&self) -> Quantity<Stress> {
        self.1.hardening_slope()
    }
}

impl<C1, C2, Y2> Viscoplastic<Y2> for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Viscoplastic<Y2>,
    Y2: Differentiate + Tensor,
{
    fn initial_state(&self) -> ViscoplasticStateVariables<Y2> {
        self.1.initial_state()
    }
    fn plastic_evolution(
        &self,
        mandel_stress: MandelStressElastic,
        state_variables: &ViscoplasticStateVariables<Y2>,
    ) -> Result<ViscoplasticEvolution<Y2>, ConstitutiveError> {
        self.1.plastic_evolution(mandel_stress, state_variables)
    }
    fn plastic_stretching_rate(
        &self,
        deviatoric_mandel_stress: MandelStressElastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<StretchingRatePlastic, ConstitutiveError> {
        self.1
            .plastic_stretching_rate(deviatoric_mandel_stress, yield_stress)
    }
    fn dissipation_potential(
        &self,
        plastic_stretching_rate: StretchingRatePlastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        self.1
            .dissipation_potential(plastic_stretching_rate, yield_stress)
    }
    fn dual_dissipation_potential(
        &self,
        deviatoric_mandel_stress: MandelStressElastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        self.1
            .dual_dissipation_potential(deviatoric_mandel_stress, yield_stress)
    }
    fn rate_sensitivity(&self) -> Scalar {
        self.1.rate_sensitivity()
    }
    fn reference_flow_rate(&self) -> Quantity<Rate> {
        self.1.reference_flow_rate()
    }
}

impl<C1, C2> ElasticPlasticOrViscoplastic for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Plastic,
{
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyStress, ConstitutiveError> {
        self.0
            .cauchy_stress(&(deformation_gradient * deformation_gradient_p.inverse()).into())
    }
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        Ok(
            CauchyTangentStiffnessElastic::from(self.0.cauchy_tangent_stiffness(
                &(deformation_gradient * &deformation_gradient_p_inverse).into(),
            )?) * deformation_gradient_p_inverse.transpose(),
        )
    }
    fn first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        Ok(
            FirstPiolaKirchhoffStressElastic::from(self.0.first_piola_kirchhoff_stress(
                &(deformation_gradient * &deformation_gradient_p_inverse).into(),
            )?) * deformation_gradient_p_inverse.transpose(),
        )
    }
    fn first_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        let deformation_gradient_p_inverse_transpose = deformation_gradient_p_inverse.transpose();
        Ok((FirstPiolaKirchhoffTangentStiffnessElastic::from(
            self.0.first_piola_kirchhoff_tangent_stiffness(
                &(deformation_gradient * &deformation_gradient_p_inverse).into(),
            )?,
        ) * &deformation_gradient_p_inverse_transpose)
            .contract_second_with_first(&deformation_gradient_p_inverse_transpose))
    }
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        Ok(&deformation_gradient_p_inverse
            * SecondPiolaKirchhoffStressElastic::from(self.0.second_piola_kirchhoff_stress(
                &(deformation_gradient * &deformation_gradient_p_inverse).into(),
            )?)
            * deformation_gradient_p_inverse.transpose())
    }
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        Ok((SecondPiolaKirchhoffTangentStiffnessElastic::from(
            self.0.second_piola_kirchhoff_tangent_stiffness(
                &(deformation_gradient * &deformation_gradient_p_inverse).into(),
            )?,
        ) * deformation_gradient_p_inverse.transpose())
        .contract_first_second_with_second(
            &deformation_gradient_p_inverse,
            &deformation_gradient_p_inverse,
        ))
    }
}

impl<C1, C2, Y2> ElasticViscoplastic<Y2> for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Viscoplastic<Y2>,
    Y2: Differentiate + Tensor,
{
}

/// The internal state `(F_p, ε_p)` evolves as `F_p` on the unimodular group
/// (`Reference → Intermediate`, so its algebra element `D_p Δt` is
/// `Intermediate → Intermediate`) and `ε_p` additively. The rate is the plastic
/// stretching rate itself, `(D_p, |D_p|)`, driven by the total deformation
/// gradient through the Mandel stress.
impl<C1, C2> StateEvolution<Time> for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Viscoplastic<Quantity>,
{
    type Field = Product<Unimodular<Intermediate, Reference>, Flat<Quantity>>;
    type Drive = DeformationGradient;
    fn initial_state(&self) -> ViscoplasticStateVariables<Quantity> {
        <Self as Viscoplastic<Quantity>>::initial_state(self)
    }
    fn state_rate(
        &self,
        _time: Quantity<Time>,
        deformation_gradient: &DeformationGradient,
        state: &ViscoplasticStateVariables<Quantity>,
    ) -> Result<ViscoplasticAlgebraRate, String> {
        let deviatoric_mandel_stress = self
            .mandel_stress(deformation_gradient, &state.0)?
            .deviatoric();
        let plastic_stretching_rate =
            self.plastic_stretching_rate(deviatoric_mandel_stress, self.yield_stress(state.1)?)?;
        let equivalent_plastic_strain_rate = plastic_stretching_rate.norm();
        Ok(TensorTuple(
            plastic_stretching_rate,
            equivalent_plastic_strain_rate,
        ))
    }
}
