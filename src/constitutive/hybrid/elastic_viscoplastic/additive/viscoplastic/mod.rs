use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::{
            plastic::Plastic,
            viscoplastic::{Viscoplastic, ViscoplasticStateVariables},
        },
        hybrid::ElasticViscoplasticAdditiveViscoplastic,
        solid::elastic_viscoplastic::ElasticViscoplastic,
    },
    math::{Rank2, Tensor, TensorTuple},
    mechanics::{MandelStressElastic, Scalar},
};

type GroupedViscoplasticStateVariables<Y1, Y2> = TensorTuple<ViscoplasticStateVariables<Y1>, Y2>;
type NestedViscoplasticStateVariables<Y1, Y2> =
    ViscoplasticStateVariables<GroupedViscoplasticStateVariables<Y1, Y2>>;

impl<C1, C2, Y1, Y2> Plastic for ElasticViscoplasticAdditiveViscoplastic<C1, C2, Y1, Y2>
where
    C1: ElasticViscoplastic<Y1>,
    C2: Viscoplastic<Y2>,
    Y1: Tensor,
    Y2: Tensor,
{
    fn initial_yield_stress(&self) -> Scalar {
        self.1.initial_yield_stress()
    }
    fn hardening_slope(&self) -> Scalar {
        self.1.hardening_slope()
    }
}

impl<C1, C2, Y1, Y2> Viscoplastic<GroupedViscoplasticStateVariables<Y1, Y2>>
    for ElasticViscoplasticAdditiveViscoplastic<C1, C2, Y1, Y2>
where
    C1: ElasticViscoplastic<Y1>,
    C2: Viscoplastic<Y2>,
    Y1: Tensor,
    Y2: Tensor,
{
    fn initial_state(&self) -> NestedViscoplasticStateVariables<Y1, Y2> {
        let initial_state_1 = self.0.initial_state();
        let (deformation_gradient, y_2) = self.1.initial_state().into();
        (deformation_gradient, (initial_state_1, y_2).into()).into()
    }
    fn plastic_evolution(
        &self,
        mandel_stress: MandelStressElastic,
        state_variables: &ViscoplasticStateVariables<GroupedViscoplasticStateVariables<Y1, Y2>>,
    ) -> Result<NestedViscoplasticStateVariables<Y1, Y2>, ConstitutiveError> {
        let state_variables_1 = &state_variables.1.0;
        let state_variables_2 = &(state_variables.0.clone(), state_variables.1.1.clone()).into();
        let deformation_gradient = (&state_variables.0).into();
        let deformation_gradient_p = &state_variables_1.0;
        let jacobian = self.0.jacobian(deformation_gradient)?;
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let cauchy_stress_1 = self
            .0
            .cauchy_stress(deformation_gradient, deformation_gradient_p)?;
        let mandel_stress_1 = (deformation_gradient_e.transpose()
            * &cauchy_stress_1
            * deformation_gradient_e.inverse_transpose())
            * jacobian;
        let cauchy_stress_2 = mandel_stress - MandelStressElastic::from(cauchy_stress_1);
        let evolution_1 = self
            .0
            .plastic_evolution(mandel_stress_1, state_variables_1)?;
        let (deformation_gradient_rate, dydt_2) = self
            .1
            .plastic_evolution(cauchy_stress_2, state_variables_2)?
            .into();
        Ok((deformation_gradient_rate, (evolution_1, dydt_2).into()).into())
    }
    fn rate_sensitivity(&self) -> Scalar {
        self.1.rate_sensitivity()
    }
    fn reference_flow_rate(&self) -> Scalar {
        self.1.reference_flow_rate()
    }
}
