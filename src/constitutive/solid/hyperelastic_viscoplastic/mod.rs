//! Hyperelastic-viscoplastic solid constitutive models.
//!
//! ---
//!
#![doc = include_str!("doc.md")]

#[cfg(feature = "doc")]
pub mod doc;

mod canonical;

use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::viscoplastic::{
            ViscoplasticEvolutionHistory, ViscoplasticStateVariables,
            ViscoplasticStateVariablesHistory,
        },
        solid::{
            elastic_plastic::bcs,
            elastic_viscoplastic::{AppliedLoad, ElasticViscoplastic},
        },
    },
    math::{
        Derivative, Differentiate, Quantity, Scalar, Tensor, TensorArray, TensorVec, Vector,
        integrate::{
            ButcherTableau, EmbeddedTableau, EvolvedIncrement, ExplicitDaeFirstOrderMinimize,
            ExplicitDaeSecondOrderMinimize, IntegrableField, StateEvolution,
            integrate_rkmk_dae_adaptive_second_order_minimize, rkmk_dae_step_second_order_minimize,
        },
        optimize::{EqualityConstraint, FirstOrderOptimization, SecondOrderOptimization},
    },
    mechanics::{
        DeformationGradient, DeformationGradientPlastic, DeformationGradients,
        FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness, Times,
    },
    units::{EnergyDensity, Time},
};
use std::ops::Mul;

/// Required methods for hyperelastic-viscoplastic solid constitutive models.
pub trait HyperelasticViscoplastic<Y>
where
    Self: ElasticViscoplastic<Y>,
    Y: Differentiate + Tensor,
{
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a = a(\mathbf{F}_\mathrm{e})
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError>;
}

/// First-order minimization methods for hyperelastic-viscoplastic solid constitutive models.
pub trait FirstOrderMinimize<Y>
where
    Y: Differentiate + Tensor,
{
    /// Solve for the unknown components of the deformation gradients under an applied load.
    ///
    /// ```math
    /// \Pi(\mathbf{F},\mathbf{F}_\mathrm{p},\boldsymbol{\lambda}) = a(\mathbf{F},\mathbf{F}_\mathrm{p}) - \boldsymbol{\lambda}:(\mathbf{F} - \mathbf{F}_0) - \mathbf{P}_0:\mathbf{F}
    /// ```
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ExplicitDaeFirstOrderMinimize<
            Quantity<EnergyDensity>,
            FirstPiolaKirchhoffStress,
            ViscoplasticStateVariables<Y>,
            DeformationGradient,
            ViscoplasticStateVariablesHistory<Y>,
            DeformationGradients,
            ViscoplasticEvolutionHistory<Y>,
        >,
        solver: impl FirstOrderOptimization<
            Quantity<EnergyDensity>,
            FirstPiolaKirchhoffStress,
            DeformationGradient,
        >,
    ) -> Result<
        (
            Times,
            DeformationGradients,
            ViscoplasticStateVariablesHistory<Y>,
        ),
        ConstitutiveError,
    >;
}

/// Second-order minimization methods for hyperelastic-viscoplastic solid constitutive models.
pub trait SecondOrderMinimize<Y>
where
    Y: Differentiate + Tensor,
{
    /// Solve for the unknown components of the deformation gradients under an applied load.
    ///
    /// ```math
    /// \Pi(\mathbf{F},\mathbf{F}_\mathrm{p},\boldsymbol{\lambda}) = a(\mathbf{F},\mathbf{F}_\mathrm{p}) - \boldsymbol{\lambda}:(\mathbf{F} - \mathbf{F}_0) - \mathbf{P}_0:\mathbf{F}
    /// ```
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ExplicitDaeSecondOrderMinimize<
            Quantity<EnergyDensity>,
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            ViscoplasticStateVariables<Y>,
            DeformationGradient,
            ViscoplasticStateVariablesHistory<Y>,
            DeformationGradients,
            ViscoplasticEvolutionHistory<Y>,
        >,
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
    >;
}

impl<C, Y> FirstOrderMinimize<Y> for C
where
    C: HyperelasticViscoplastic<Y>,
    Y: Differentiate + Tensor,
{
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ExplicitDaeFirstOrderMinimize<
            Quantity<EnergyDensity>,
            FirstPiolaKirchhoffStress,
            ViscoplasticStateVariables<Y>,
            DeformationGradient,
            ViscoplasticStateVariablesHistory<Y>,
            DeformationGradients,
            ViscoplasticEvolutionHistory<Y>,
        >,
        solver: impl FirstOrderOptimization<
            Quantity<EnergyDensity>,
            FirstPiolaKirchhoffStress,
            DeformationGradient,
        >,
    ) -> Result<
        (
            Times,
            DeformationGradients,
            ViscoplasticStateVariablesHistory<Y>,
        ),
        ConstitutiveError,
    > {
        let (matrix, prescribed, time) = bcs(applied_load);
        let mut vector = Vector::zero(matrix.len());
        let (times, state_variables, _, deformation_gradients) = integrator
            .integrate(
                |_: Quantity<Time>,
                 state_variables: &ViscoplasticStateVariables<Y>,
                 deformation_gradient: &DeformationGradient| {
                    Ok(self.state_variables_evolution(deformation_gradient, state_variables)?)
                },
                |_: Quantity<Time>,
                 state_variables: &ViscoplasticStateVariables<Y>,
                 deformation_gradient: &DeformationGradient| {
                    let deformation_gradient_p = &state_variables.0;
                    Ok(self.helmholtz_free_energy_density(
                        deformation_gradient,
                        deformation_gradient_p,
                    )?)
                },
                |_: Quantity<Time>,
                 state_variables: &ViscoplasticStateVariables<Y>,
                 deformation_gradient: &DeformationGradient| {
                    let deformation_gradient_p = &state_variables.0;
                    Ok(self.first_piola_kirchhoff_stress(
                        deformation_gradient,
                        deformation_gradient_p,
                    )?)
                },
                solver,
                time,
                (self.initial_state(), DeformationGradient::identity()),
                |t: Quantity<Time>| {
                    prescribed
                        .iter()
                        .for_each(|(index, function)| vector[*index] = function(t));
                    EqualityConstraint::Linear(matrix.clone(), vector.clone())
                },
            )
            .map_err(|error| ConstitutiveError::upstream(error, self))?;
        Ok((times, deformation_gradients, state_variables))
    }
}

impl<C, Y> SecondOrderMinimize<Y> for C
where
    C: HyperelasticViscoplastic<Y>,
    Y: Differentiate + Tensor,
{
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ExplicitDaeSecondOrderMinimize<
            Quantity<EnergyDensity>,
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            ViscoplasticStateVariables<Y>,
            DeformationGradient,
            ViscoplasticStateVariablesHistory<Y>,
            DeformationGradients,
            ViscoplasticEvolutionHistory<Y>,
        >,
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
    > {
        let (matrix, prescribed, time) = bcs(applied_load);
        let mut vector = Vector::zero(matrix.len());
        let (times, state_variables, _, deformation_gradients) = integrator
            .integrate(
                |_: Quantity<Time>,
                 state_variables: &ViscoplasticStateVariables<Y>,
                 deformation_gradient: &DeformationGradient| {
                    Ok(self.state_variables_evolution(deformation_gradient, state_variables)?)
                },
                |_: Quantity<Time>,
                 state_variables: &ViscoplasticStateVariables<Y>,
                 deformation_gradient: &DeformationGradient| {
                    let deformation_gradient_p = &state_variables.0;
                    Ok(self.helmholtz_free_energy_density(
                        deformation_gradient,
                        deformation_gradient_p,
                    )?)
                },
                |_: Quantity<Time>,
                 state_variables: &ViscoplasticStateVariables<Y>,
                 deformation_gradient: &DeformationGradient| {
                    let deformation_gradient_p = &state_variables.0;
                    Ok(self.first_piola_kirchhoff_stress(
                        deformation_gradient,
                        deformation_gradient_p,
                    )?)
                },
                |_: Quantity<Time>,
                 state_variables: &ViscoplasticStateVariables<Y>,
                 deformation_gradient: &DeformationGradient| {
                    let deformation_gradient_p = &state_variables.0;
                    Ok(self.first_piola_kirchhoff_tangent_stiffness(
                        deformation_gradient,
                        deformation_gradient_p,
                    )?)
                },
                solver,
                time,
                (self.initial_state(), DeformationGradient::identity()),
                |t: Quantity<Time>| {
                    prescribed
                        .iter()
                        .for_each(|(index, function)| vector[*index] = function(t));
                    EqualityConstraint::Linear(matrix.clone(), vector.clone())
                },
                None,
            )
            .map_err(|error| ConstitutiveError::upstream(error, self))?;
        Ok((times, deformation_gradients, state_variables))
    }
}

/// RKMK-DAE return-map methods for hyperelastic-viscoplastic solid
/// constitutive models — the minimize-based sibling of
/// [`crate::constitutive::solid::elastic_viscoplastic::RootRkmkDae`], for
/// models whose equilibrium is naturally posed as a potential minimization
/// rather than a stress-residual root. `F_p` still advances on its group at
/// the tableau's own order; only how `F` is resolved within a stage differs.
/// Blanket over any [`HyperelasticViscoplastic`] model, same as
/// [`SecondOrderMinimize`] itself.
pub trait RootRkmkDaeMinimize<Y>
where
    Y: Differentiate + Tensor,
{
    /// `F` is re-solved by potential minimization at every stage abscissa of
    /// the window while `F_p` advances on its group, generic over the
    /// hardening variable `Y`.
    fn root_rkmk_dae_minimize<Tab: ButcherTableau>(
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
    >;
    /// As [`Self::root_rkmk_dae_minimize`], but the whole span is stepped
    /// under embedded (`Tab::D`) error control rather than on the supplied
    /// load grid.
    fn root_rkmk_dae_adaptive_minimize<Tab: EmbeddedTableau>(
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
    >;
}

impl<C, Y> RootRkmkDaeMinimize<Y> for C
where
    C: HyperelasticViscoplastic<Y>
        + StateEvolution<
            Time,
            Y,
            Drive = DeformationGradient,
            Field: IntegrableField<Point = ViscoplasticStateVariables<Y>>,
        >,
    Y: Differentiate + Tensor,
    EvolvedIncrement<C, Time, Y>: Clone + Differentiate<Time>,
    for<'a> &'a Derivative<EvolvedIncrement<C, Time, Y>, Time>:
        Mul<Quantity<Time>, Output = EvolvedIncrement<C, Time, Y>>,
{
    #[allow(clippy::type_complexity)]
    fn root_rkmk_dae_minimize<Tab: ButcherTableau>(
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
    > {
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
    #[allow(clippy::type_complexity)]
    fn root_rkmk_dae_adaptive_minimize<Tab: EmbeddedTableau>(
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
    > {
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
