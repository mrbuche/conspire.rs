#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        canonical::Canonical,
        fluid::viscoplastic::{
            Viscoplastic, ViscoplasticStateVariables, ViscoplasticStateVariablesHistory,
        },
        solid::{
            elastic_plastic::bcs,
            elastic_viscoplastic::{AppliedLoad, ElasticPlasticOrViscoplastic},
            hyperelastic::Hyperelastic,
            hyperelastic_viscoplastic::HyperelasticViscoplastic,
        },
    },
    math::{
        Derivative, Differentiate, Quantity, Scalar, Tensor, TensorArray, TensorVec, Vector,
        integrate::{
            ButcherTableau, EmbeddedTableau, EvolvedIncrement, IntegrableField, StateEvolution,
            integrate_rkmk_dae_adaptive_second_order_minimize, rkmk_dae_step_second_order_minimize,
        },
        optimize::{EqualityConstraint, SecondOrderOptimization},
    },
    mechanics::{
        DeformationGradient, DeformationGradientPlastic, DeformationGradients,
        FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness, Times,
    },
    units::{EnergyDensity, Time},
};
use std::ops::Mul;

impl<C1, C2, Y2> HyperelasticViscoplastic<Y2> for Canonical<C1, C2>
where
    C1: Hyperelastic,
    C2: Viscoplastic<Y2>,
    Y2: Differentiate + Tensor,
{
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        self.0
            .helmholtz_free_energy_density(&deformation_gradient_e.into())
    }
}

impl<C1, C2> Canonical<C1, C2>
where
    C1: Hyperelastic,
{
    /// As [`crate::constitutive::solid::elastic_viscoplastic::Canonical::root_rkmk_dae`],
    /// but the equilibrium leg at every stage abscissa is a potential
    /// minimization rather than a stress-residual root — the sibling built by
    /// [`rkmk_dae_step_second_order_minimize`] for models whose equilibrium is
    /// naturally posed that way. `F_p` still advances on its group at the
    /// tableau's own order; only how `F` is resolved within a stage differs.
    #[allow(clippy::type_complexity)]
    pub fn root_rkmk_dae_minimize<Tab, Y>(
        &self,
        applied_load: AppliedLoad,
        solver: impl SecondOrderOptimization<
            Quantity<EnergyDensity>,
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
            + HyperelasticViscoplastic<Y>
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
         -> Result<Quantity<EnergyDensity>, String> {
            Ok(self.helmholtz_free_energy_density(deformation_gradient, &state.0)?)
        };
        let jacobian = |_: Quantity<Time>,
                        state: &ViscoplasticStateVariables<Y>,
                        deformation_gradient: &DeformationGradient|
         -> Result<FirstPiolaKirchhoffStress, String> {
            Ok(self.first_piola_kirchhoff_stress(deformation_gradient, &state.0)?)
        };
        let hessian = |_: Quantity<Time>,
                       state: &ViscoplasticStateVariables<Y>,
                       deformation_gradient: &DeformationGradient|
         -> Result<FirstPiolaKirchhoffTangentStiffness, String> {
            Ok(self.first_piola_kirchhoff_tangent_stiffness(deformation_gradient, &state.0)?)
        };
        let mut deformation_gradient = solver
            .minimize(
                |deformation_gradient: &DeformationGradient| {
                    function(time[0], &state, deformation_gradient)
                },
                |deformation_gradient: &DeformationGradient| {
                    jacobian(time[0], &state, deformation_gradient)
                },
                |deformation_gradient: &DeformationGradient| {
                    hessian(time[0], &state, deformation_gradient)
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
            let advanced = rkmk_dae_step_second_order_minimize::<
                <Self as StateEvolution<Time, Y>>::Field,
                Tab,
                Quantity<EnergyDensity>,
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
                hessian,
                &solver,
                &state,
                &deformation_gradient,
                step[0],
                step[1] - step[0],
                &mut scratch,
                carry.as_ref(),
                equality_constraint,
                None,
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
    /// As [`Self::root_rkmk_dae_minimize`], but the whole span is stepped
    /// under embedded (`Tab::D`) error control rather than on the supplied
    /// load grid — the minimize-based sibling of
    /// [`crate::constitutive::solid::elastic_viscoplastic::Canonical::root_rkmk_dae_adaptive`].
    #[allow(clippy::type_complexity)]
    pub fn root_rkmk_dae_adaptive_minimize<Tab, Y>(
        &self,
        applied_load: AppliedLoad,
        solver: impl SecondOrderOptimization<
            Quantity<EnergyDensity>,
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
            + HyperelasticViscoplastic<Y>
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
         -> Result<Quantity<EnergyDensity>, String> {
            Ok(self.helmholtz_free_energy_density(deformation_gradient, &state.0)?)
        };
        let jacobian = |_: Quantity<Time>,
                        state: &ViscoplasticStateVariables<Y>,
                        deformation_gradient: &DeformationGradient|
         -> Result<FirstPiolaKirchhoffStress, String> {
            Ok(self.first_piola_kirchhoff_stress(deformation_gradient, &state.0)?)
        };
        let hessian = |_: Quantity<Time>,
                       state: &ViscoplasticStateVariables<Y>,
                       deformation_gradient: &DeformationGradient|
         -> Result<FirstPiolaKirchhoffTangentStiffness, String> {
            Ok(self.first_piola_kirchhoff_tangent_stiffness(deformation_gradient, &state.0)?)
        };
        let deformation_gradient = solver
            .minimize(
                |deformation_gradient: &DeformationGradient| {
                    function(time[0], &state, deformation_gradient)
                },
                |deformation_gradient: &DeformationGradient| {
                    jacobian(time[0], &state, deformation_gradient)
                },
                |deformation_gradient: &DeformationGradient| {
                    hessian(time[0], &state, deformation_gradient)
                },
                DeformationGradient::identity(),
                equality_constraint(time[0]),
                None,
            )
            .map_err(|error| ConstitutiveError::upstream(String::from(error), self))?;
        let (times, state_variables, deformation_gradients) =
            integrate_rkmk_dae_adaptive_second_order_minimize::<
                <Self as StateEvolution<Time, Y>>::Field,
                Tab,
                Quantity<EnergyDensity>,
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
                hessian,
                &solver,
                time,
                (state, deformation_gradient),
                abs_tol,
                rel_tol,
                equality_constraint,
                None,
            )
            .map_err(|error| ConstitutiveError::upstream(error, self))?;
        Ok((times, deformation_gradients, state_variables))
    }
}
