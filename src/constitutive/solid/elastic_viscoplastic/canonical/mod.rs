#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        canonical::Canonical,
        fluid::{
            plastic::Plastic,
            viscoplastic::{Viscoplastic, ViscoplasticEvolution, ViscoplasticStateVariables},
        },
        solid::{
            elastic::Elastic,
            elastic_viscoplastic::{
                ElasticPlasticOrViscoplastic, ElasticViscoplastic, PlasticTangents,
            },
        },
    },
    math::{
        ContractFirstSecondWithSecond, ContractSecondWithFirst, ContractThirdWithFirst, Derivative,
        Differentiate, Intermediate, Quantity, Rank2, Reference, Scalar, Tensor, TensorRank2,
        TensorTuple,
        integrate::{Flat, IntegrableField, Product, StateEvolution, Unimodular},
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, CauchyTangentStiffnessElastic,
        CauchyTangentStiffnessPlastic, DeformationGradient, DeformationGradientPlastic,
        FirstPiolaKirchhoffStress, FirstPiolaKirchhoffStressElastic,
        FirstPiolaKirchhoffTangentStiffness, FirstPiolaKirchhoffTangentStiffnessElastic,
        FirstPiolaKirchhoffTangentStiffnessPlastic, MandelStressElastic,
        SecondPiolaKirchhoffStress, SecondPiolaKirchhoffStressElastic,
        SecondPiolaKirchhoffTangentStiffness, SecondPiolaKirchhoffTangentStiffnessElastic,
        StretchingRatePlastic,
    },
    units::{Dissipation, Rate, Stress, Time},
};
use std::ops::Add;

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

impl<C1, C2> PlasticTangents for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Plastic,
{
    fn cauchy_tangent_stiffness_p(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyTangentStiffnessPlastic, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        let deformation_gradient_e = deformation_gradient * &deformation_gradient_p_inverse;
        Ok(CauchyTangentStiffnessElastic::from(
            self.0
                .cauchy_tangent_stiffness(&deformation_gradient_e.clone().into())?,
        )
        .contract_third_with_first(&deformation_gradient_e)
            * deformation_gradient_p_inverse.transpose()
            * -1.0)
    }
    fn first_piola_kirchhoff_tangent_stiffness_p(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffTangentStiffnessPlastic, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        let deformation_gradient_p_inverse_transpose = deformation_gradient_p_inverse.transpose();
        let deformation_gradient_e = deformation_gradient * &deformation_gradient_p_inverse;
        let first_piola_kirchhoff_stress =
            self.first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?;
        Ok(((FirstPiolaKirchhoffTangentStiffnessElastic::from(
            self.0
                .first_piola_kirchhoff_tangent_stiffness(&deformation_gradient_e.clone().into())?,
        )
        .contract_third_with_first(&deformation_gradient_e)
            * &deformation_gradient_p_inverse_transpose)
            .contract_second_with_first(&deformation_gradient_p_inverse_transpose)
            + FirstPiolaKirchhoffTangentStiffnessPlastic::dyad_il_kj(
                &first_piola_kirchhoff_stress,
                &deformation_gradient_p_inverse_transpose,
            ))
            * -1.0)
    }
}

impl<C1, C2, Y2> ElasticViscoplastic<Y2> for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Viscoplastic<Y2>,
    Y2: Differentiate + Tensor,
{
}

/// The internal state `(F_p, Y)` evolves as `F_p` on the unimodular group
/// (`Reference → Intermediate`, so its algebra element `D_p Δt` is
/// `Intermediate → Intermediate`) and the hardening variable `Y` additively. The
/// rate is `(D_p, Ẏ)`, from the model's [`plastic_evolution`] (`D_p` recovered as
/// `Ḟ_p F_p⁻¹`), driven by the total deformation gradient through the Mandel
/// stress.
///
/// [`plastic_evolution`]: Viscoplastic::plastic_evolution
impl<C1, C2, Y> StateEvolution<Time, Y> for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Viscoplastic<Y>,
    Y: Clone + Differentiate<Time> + Tensor,
    for<'a> Y: Add<&'a Y, Output = Y>,
    TensorTuple<TensorRank2<3, Intermediate, Intermediate>, Y>: Differentiate<
            Time,
            Derivative = TensorTuple<
                TensorRank2<3, Intermediate, Intermediate, Rate>,
                Derivative<Y>,
            >,
        >,
{
    type Field = Product<Unimodular<Intermediate, Reference>, Flat<Y>>;
    type Drive = DeformationGradient;
    fn initial_state(&self) -> ViscoplasticStateVariables<Y> {
        <Self as Viscoplastic<Y>>::initial_state(self)
    }
    fn state_rate(
        &self,
        _time: Quantity<Time>,
        deformation_gradient: &DeformationGradient,
        state: &ViscoplasticStateVariables<Y>,
    ) -> Result<Derivative<<Self::Field as IntegrableField>::Increment, Time>, String> {
        let mandel_stress = self.mandel_stress(deformation_gradient, &state.0)?;
        let evolution = self.plastic_evolution(mandel_stress, state)?;
        let plastic_stretching_rate = evolution.0 * state.0.inverse();
        Ok(TensorTuple(plastic_stretching_rate, evolution.1))
    }
}
