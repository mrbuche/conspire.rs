#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        canonical::Canonical,
        fluid::{
            plastic::Plastic,
            viscoplastic::{
                Viscoplastic, ViscoplasticEvolution, ViscoplasticStateVariables,
                ViscoplasticStateVariablesHistory,
            },
        },
        solid::{
            elastic::Elastic,
            elastic_plastic::bcs,
            elastic_viscoplastic::{
                AppliedLoad, ElasticPlasticOrViscoplastic, ElasticViscoplastic,
            },
        },
    },
    math::{
        ContractFirstSecondWithSecond, ContractSecondWithFirst, Derivative, Differentiate,
        Intermediate, Quantity, Rank2, Reference, Scalar, Tensor, TensorArray, TensorRank2,
        TensorTuple, TensorVec, Vector,
        integrate::{
            ButcherTableau, EmbeddedTableau, EvolvedIncrement, Flat, IntegrableField, Product,
            StateEvolution, Unimodular, integrate_rkmk_dae_adaptive_first_order_root,
            rkmk_dae_step_first_order_root,
        },
        optimize::{EqualityConstraint, FirstOrderRootFinding},
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, CauchyTangentStiffnessElastic, DeformationGradient,
        DeformationGradientPlastic, DeformationGradients, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffStressElastic, FirstPiolaKirchhoffTangentStiffness,
        FirstPiolaKirchhoffTangentStiffnessElastic, MandelStressElastic,
        SecondPiolaKirchhoffStress, SecondPiolaKirchhoffStressElastic,
        SecondPiolaKirchhoffTangentStiffness, SecondPiolaKirchhoffTangentStiffnessElastic,
        StretchingRatePlastic, Times,
    },
    units::{Dissipation, Rate, Stress, Time},
};
use std::ops::{Add, Mul};

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

impl<C1, C2> Canonical<C1, C2>
where
    C1: Elastic,
{
    /// RKMK-DAE return map: `F` is re-solved from equilibrium at every stage
    /// abscissa of the window while `F_p` advances on its group, generic over
    /// the hardening variable `Y`. The tableau drives both legs: stage `i`
    /// reconstructs `F_p` at `σᵢ`, solves
    /// `P(F, F_p^i) - λ(t + cᵢ Δt) - P_0 = 0` for `F` there, and evaluates the
    /// plastic rate at that consistent pair, so the order is the tableau's own
    /// (unlike a scheme that freezes `F` across the whole window, which would
    /// cap the coupling at first order regardless of the tableau).
    ///
    /// This is the half-explicit RK treatment of the index-1 DAE that
    /// [`FirstOrderRoot::root`] already performs, with the state leg moved off
    /// the additive march onto `expm`/`dexpinv` — so `det F_p = 1` is kept
    /// rather than drifting.
    ///
    /// [`FirstOrderRoot::root`]: crate::constitutive::solid::elastic_viscoplastic::FirstOrderRoot::root
    #[allow(clippy::type_complexity)]
    pub fn root_rkmk_dae<Tab, Y>(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
    ) -> Result<
        (
            Times,
            DeformationGradients,
            ViscoplasticStateVariablesHistory<Y>,
        ),
        ConstitutiveError,
    >
    where
        Tab: ButcherTableau,
        C2: Viscoplastic<Y>,
        Y: Differentiate + Tensor,
        Self: ElasticPlasticOrViscoplastic
            + Viscoplastic<Y>
            + StateEvolution<
                Time,
                Y,
                Drive = DeformationGradient,
                Field: IntegrableField<Point = ViscoplasticStateVariables<Y>>,
            >,
        EvolvedIncrement<Self, Time, Y>: Clone + Differentiate<Time>,
        for<'a> &'a Derivative<EvolvedIncrement<Self, Time, Y>, Time>:
            Mul<Quantity<Time>, Output = EvolvedIncrement<Self, Time, Y>>,
    {
        let (matrix, prescribed, time) = bcs(applied_load);
        let mut state = <Self as StateEvolution<Time, Y>>::initial_state(self);
        let mut scratch = Vec::new();
        let equality_constraint = |t: Quantity<Time>| {
            let mut vector = Vector::zero(matrix.len());
            prescribed
                .iter()
                .for_each(|(index, function)| vector[*index] = function(t));
            EqualityConstraint::Linear(matrix.clone(), vector)
        };
        let function = |_: Quantity<Time>,
                        state: &ViscoplasticStateVariables<Y>,
                        deformation_gradient: &DeformationGradient|
         -> Result<FirstPiolaKirchhoffStress, String> {
            Ok(self.first_piola_kirchhoff_stress(deformation_gradient, &state.0)?)
        };
        let jacobian = |_: Quantity<Time>,
                        state: &ViscoplasticStateVariables<Y>,
                        deformation_gradient: &DeformationGradient|
         -> Result<FirstPiolaKirchhoffTangentStiffness, String> {
            Ok(self.first_piola_kirchhoff_tangent_stiffness(deformation_gradient, &state.0)?)
        };
        let mut deformation_gradient = solver
            .root(
                |deformation_gradient: &DeformationGradient| {
                    function(time[0], &state, deformation_gradient)
                },
                |deformation_gradient: &DeformationGradient| {
                    jacobian(time[0], &state, deformation_gradient)
                },
                DeformationGradient::identity(),
                equality_constraint(time[0]),
                None,
            )
            .map_err(|error| ConstitutiveError::upstream(String::from(error), self))?;
        let mut times = Times::new();
        let mut deformation_gradients = DeformationGradients::new();
        let mut state_variables = ViscoplasticStateVariablesHistory::new();
        let mut carry = None;
        times.push(time[0]);
        deformation_gradients.push(deformation_gradient.clone());
        state_variables.push(state.clone());
        for step in time.windows(2) {
            let advanced = rkmk_dae_step_first_order_root::<
                <Self as StateEvolution<Time, Y>>::Field,
                Tab,
                FirstPiolaKirchhoffStress,
                FirstPiolaKirchhoffTangentStiffness,
                DeformationGradient,
                Time,
            >(
                &mut |t, state, deformation_gradient| {
                    self.state_rate(t, deformation_gradient, state)
                },
                function,
                jacobian,
                &solver,
                &state,
                &deformation_gradient,
                step[0],
                step[1] - step[0],
                &mut scratch,
                carry.as_ref(),
                equality_constraint,
            )
            .map_err(|error| ConstitutiveError::upstream(error, self))?;
            state = advanced.0;
            deformation_gradient = advanced.1;
            carry = advanced.2;
            times.push(step[1]);
            deformation_gradients.push(deformation_gradient.clone());
            state_variables.push(state.clone());
        }
        Ok((times, deformation_gradients, state_variables))
    }
    /// As [`Self::root_rkmk_dae`], but the whole span is stepped under embedded
    /// (`Tab::D`) error control rather than on the supplied load grid.
    ///
    /// Two times in `applied_load` give only the span, and the controller's own
    /// accepted steps are reported. More than two are requested report times —
    /// the convention of the flat DAE loop — and `F_p` is served at each from
    /// the geodesic [`HermiteSegment`] of the accepted step containing it, so it
    /// is on the unimodular group at every reported time and not just at the
    /// accepted ones; `F` is then re-solved from equilibrium there.
    ///
    /// [`HermiteSegment`]: crate::math::integrate::HermiteSegment
    #[allow(clippy::type_complexity)]
    pub fn root_rkmk_dae_adaptive<Tab, Y>(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
        abs_tol: Scalar,
        rel_tol: Scalar,
    ) -> Result<
        (
            Times,
            DeformationGradients,
            ViscoplasticStateVariablesHistory<Y>,
        ),
        ConstitutiveError,
    >
    where
        Tab: EmbeddedTableau,
        C2: Viscoplastic<Y>,
        Y: Differentiate + Tensor,
        Self: ElasticPlasticOrViscoplastic
            + Viscoplastic<Y>
            + StateEvolution<
                Time,
                Y,
                Drive = DeformationGradient,
                Field: IntegrableField<Point = ViscoplasticStateVariables<Y>>,
            >,
        EvolvedIncrement<Self, Time, Y>: Clone + Differentiate<Time>,
        for<'a> &'a Derivative<EvolvedIncrement<Self, Time, Y>, Time>:
            Mul<Quantity<Time>, Output = EvolvedIncrement<Self, Time, Y>>,
    {
        let (matrix, prescribed, time) = bcs(applied_load);
        let state = <Self as StateEvolution<Time, Y>>::initial_state(self);
        let equality_constraint = |t: Quantity<Time>| {
            let mut vector = Vector::zero(matrix.len());
            prescribed
                .iter()
                .for_each(|(index, function)| vector[*index] = function(t));
            EqualityConstraint::Linear(matrix.clone(), vector)
        };
        let function = |_: Quantity<Time>,
                        state: &ViscoplasticStateVariables<Y>,
                        deformation_gradient: &DeformationGradient|
         -> Result<FirstPiolaKirchhoffStress, String> {
            Ok(self.first_piola_kirchhoff_stress(deformation_gradient, &state.0)?)
        };
        let jacobian = |_: Quantity<Time>,
                        state: &ViscoplasticStateVariables<Y>,
                        deformation_gradient: &DeformationGradient|
         -> Result<FirstPiolaKirchhoffTangentStiffness, String> {
            Ok(self.first_piola_kirchhoff_tangent_stiffness(deformation_gradient, &state.0)?)
        };
        let deformation_gradient = solver
            .root(
                |deformation_gradient: &DeformationGradient| {
                    function(time[0], &state, deformation_gradient)
                },
                |deformation_gradient: &DeformationGradient| {
                    jacobian(time[0], &state, deformation_gradient)
                },
                DeformationGradient::identity(),
                equality_constraint(time[0]),
                None,
            )
            .map_err(|error| ConstitutiveError::upstream(String::from(error), self))?;
        let (times, state_variables, deformation_gradients) =
            integrate_rkmk_dae_adaptive_first_order_root::<
                <Self as StateEvolution<Time, Y>>::Field,
                Tab,
                FirstPiolaKirchhoffStress,
                FirstPiolaKirchhoffTangentStiffness,
                DeformationGradient,
                ViscoplasticStateVariablesHistory<Y>,
                DeformationGradients,
                Time,
            >(
                |t, state, deformation_gradient| self.state_rate(t, deformation_gradient, state),
                function,
                jacobian,
                &solver,
                time,
                (state, deformation_gradient),
                abs_tol,
                rel_tol,
                equality_constraint,
            )
            .map_err(|error| ConstitutiveError::upstream(error, self))?;
        Ok((times, deformation_gradients, state_variables))
    }
}
