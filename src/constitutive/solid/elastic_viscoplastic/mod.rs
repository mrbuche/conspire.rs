//! Elastic-viscoplastic solid constitutive models.
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
        canonical::Canonical,
        fluid::viscoplastic::{
            Viscoplastic, ViscoplasticEvolution, ViscoplasticEvolutionHistory,
            ViscoplasticStateVariables, ViscoplasticStateVariablesHistory,
        },
        solid::elastic::Elastic,
    },
    math::{
        ContractWith, Derivative, Differentiate, Quantity, Rank2, Tensor, TensorArray, TensorVec,
        Vector,
        integrate::{
            EmbeddedTableau, EvolvedIncrement, ExplicitDaeFirstOrderRoot,
            ExplicitDaeZerothOrderRoot, IntegrableField, StateEvolution, rkmk_step,
        },
        optimize::{EqualityConstraint, FirstOrderRootFinding, ZerothOrderRootFinding},
    },
    mechanics::{
        DeformationGradient, DeformationGradientPlastic, DeformationGradients,
        FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness, Times,
    },
    units::{Dissipation, Time},
};

use crate::constitutive::solid::elastic_plastic::bcs;
pub use crate::constitutive::solid::elastic_plastic::{AppliedLoad, ElasticPlasticOrViscoplastic};
use std::ops::Mul;

/// Required methods for elastic-viscoplastic solid constitutive models.
pub trait ElasticViscoplastic<Y>
where
    Self: ElasticPlasticOrViscoplastic + Viscoplastic<Y>,
    Y: Differentiate + Tensor,
{
    /// Calculates and returns the internal dissipation.
    ///
    /// ```math
    /// T\dot{s} = \mathbf{M}_\mathrm{e}':\mathbf{D}_\mathrm{p}
    /// ```
    fn internal_dissipation(
        &self,
        deformation_gradient: &DeformationGradient,
        state_variables: &ViscoplasticStateVariables<Y>,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        let deformation_gradient_p = &state_variables.0;
        let plastic_stretching_rate = self
            .state_variables_evolution(deformation_gradient, state_variables)?
            .0
            * deformation_gradient_p.inverse();
        Ok(self
            .mandel_stress(deformation_gradient, deformation_gradient_p)?
            .deviatoric()
            .contract_with(&plastic_stretching_rate))
    }
    /// Calculates and returns the evolution of the state variables.
    fn state_variables_evolution(
        &self,
        deformation_gradient: &DeformationGradient,
        state_variables: &ViscoplasticStateVariables<Y>,
    ) -> Result<ViscoplasticEvolution<Y>, ConstitutiveError> {
        self.plastic_evolution(
            self.mandel_stress(deformation_gradient, &state_variables.0)?,
            state_variables,
        )
    }
}

/// Zeroth-order root-finding methods for elastic-viscoplastic solid constitutive models.
pub trait ZerothOrderRoot<Y>
where
    Y: Differentiate + Tensor,
{
    /// Solve for the unknown components of the deformation gradients under an applied load.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F},\mathbf{F}_\mathrm{p}) - \boldsymbol{\lambda} - \mathbf{P}_0 = \mathbf{0}
    /// ```
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ExplicitDaeZerothOrderRoot<
            FirstPiolaKirchhoffStress,
            ViscoplasticStateVariables<Y>,
            DeformationGradient,
            ViscoplasticStateVariablesHistory<Y>,
            DeformationGradients,
            ViscoplasticEvolutionHistory<Y>,
        >,
        solver: impl ZerothOrderRootFinding<FirstPiolaKirchhoffStress, DeformationGradient>,
    ) -> Result<
        (
            Times,
            DeformationGradients,
            ViscoplasticStateVariablesHistory<Y>,
        ),
        ConstitutiveError,
    >;
}

/// First-order root-finding methods for elastic-viscoplastic solid constitutive models.
pub trait FirstOrderRoot<Y>
where
    Y: Differentiate + Tensor,
{
    /// Solve for the unknown components of the deformation gradients under an applied load.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F},\mathbf{F}_\mathrm{p}) - \boldsymbol{\lambda} - \mathbf{P}_0 = \mathbf{0}
    /// ```
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ExplicitDaeFirstOrderRoot<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            ViscoplasticStateVariables<Y>,
            DeformationGradient,
            ViscoplasticStateVariablesHistory<Y>,
            DeformationGradients,
            ViscoplasticEvolutionHistory<Y>,
        >,
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
    >;
}

impl<C, Y> ZerothOrderRoot<Y> for C
where
    C: ElasticViscoplastic<Y>,
    Y: Differentiate + Tensor,
{
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ExplicitDaeZerothOrderRoot<
            FirstPiolaKirchhoffStress,
            ViscoplasticStateVariables<Y>,
            DeformationGradient,
            ViscoplasticStateVariablesHistory<Y>,
            DeformationGradients,
            ViscoplasticEvolutionHistory<Y>,
        >,
        solver: impl ZerothOrderRootFinding<FirstPiolaKirchhoffStress, DeformationGradient>,
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

impl<C, Y> FirstOrderRoot<Y> for C
where
    C: ElasticViscoplastic<Y>,
    Y: Differentiate + Tensor,
{
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ExplicitDaeFirstOrderRoot<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            ViscoplasticStateVariables<Y>,
            DeformationGradient,
            ViscoplasticStateVariablesHistory<Y>,
            DeformationGradients,
            ViscoplasticEvolutionHistory<Y>,
        >,
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
            )
            .map_err(|error| ConstitutiveError::upstream(error, self))?;
        Ok((times, deformation_gradients, state_variables))
    }
}

/// Interim RKMK return map — an operator-split alternative to
/// [`FirstOrderRoot::root`] that advances the plastic state on its manifold
/// (`F_p` stays unimodular) instead of marching it additively.
///
/// A Lie–Trotter split: equilibrium is solved once at the initial time, then each
/// step takes one [`rkmk_step`] for `(F_p, ε_p)` with `F` frozen at the current
/// equilibrium, and re-solves `P(F, F_p) - λ - P_0 = 0` for `F` at the new time
/// with the advanced `F_p` held — so every recorded `(t, F, state)` is mutually
/// consistent. First order in the `F ↔ F_p` coupling; a monolithic RKMK return
/// map (the group state threaded through the shared DAE solver) is future work —
/// see the heterogeneous-integration notes.
pub trait RkmkRoot<Y = Quantity>
where
    Y: Differentiate + Tensor,
{
    /// Solve for the unknown deformation-gradient components under an applied
    /// load, advancing the plastic state with a `Tab`-tableau RKMK step.
    fn root_rkmk<Tab>(
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
        Tab: EmbeddedTableau;
}

impl<C1, C2, Y> RkmkRoot<Y> for Canonical<C1, C2>
where
    C1: Elastic,
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
    fn root_rkmk<Tab>(
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
        Tab: EmbeddedTableau,
    {
        let (matrix, prescribed, time) = bcs(applied_load);
        let mut vector = Vector::zero(matrix.len());
        let mut state = <Self as StateEvolution<Time, Y>>::initial_state(self);
        let mut scratch: Vec<EvolvedIncrement<Self, Time, Y>> = Vec::new();
        let mut equilibrate = |deformation_gradient_p: &DeformationGradientPlastic,
                               guess: &DeformationGradient,
                               t: Quantity<Time>|
         -> Result<DeformationGradient, ConstitutiveError> {
            prescribed
                .iter()
                .for_each(|(index, function)| vector[*index] = function(t));
            solver
                .root(
                    |deformation_gradient: &DeformationGradient| {
                        Ok(self.first_piola_kirchhoff_stress(
                            deformation_gradient,
                            deformation_gradient_p,
                        )?)
                    },
                    |deformation_gradient: &DeformationGradient| {
                        Ok(self.first_piola_kirchhoff_tangent_stiffness(
                            deformation_gradient,
                            deformation_gradient_p,
                        )?)
                    },
                    guess.clone(),
                    EqualityConstraint::Linear(matrix.clone(), vector.clone()),
                    None,
                )
                .map_err(|error| ConstitutiveError::upstream(error, self))
        };
        let deformation_gradient_p = state.0.clone();
        let mut deformation_gradient = equilibrate(
            &deformation_gradient_p,
            &DeformationGradient::identity(),
            time[0],
        )?;
        let mut times = Times::new();
        let mut deformation_gradients = DeformationGradients::new();
        let mut state_variables = ViscoplasticStateVariablesHistory::new();
        times.push(time[0]);
        deformation_gradients.push(deformation_gradient.clone());
        state_variables.push(state.clone());
        for step in time.windows(2) {
            let frozen = deformation_gradient.clone();
            state = rkmk_step::<<Self as StateEvolution<Time, Y>>::Field, Tab, Time>(
                &mut |t, point| self.state_rate(t, &frozen, point),
                &state,
                step[0],
                step[1] - step[0],
                &mut scratch,
            )
            .map_err(|error| ConstitutiveError::upstream(error, self))?;
            let deformation_gradient_p = state.0.clone();
            deformation_gradient =
                equilibrate(&deformation_gradient_p, &deformation_gradient, step[1])?;
            times.push(step[1]);
            deformation_gradients.push(deformation_gradient.clone());
            state_variables.push(state.clone());
        }
        Ok((times, deformation_gradients, state_variables))
    }
}
