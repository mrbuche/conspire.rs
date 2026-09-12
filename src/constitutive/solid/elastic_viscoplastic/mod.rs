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
        fluid::viscoplastic::{
            Viscoplastic, ViscoplasticEvolution, ViscoplasticEvolutionHistory,
            ViscoplasticStateVariables, ViscoplasticStateVariablesHistory,
        },
    },
    math::{
        ContractWith, Derivative, Differentiate, Intermediate, Quantity, Rank2, Reference, Scalar,
        Tensor, TensorArray, TensorRank2, TensorTuple, TensorVec, Vector,
        integrate::{
            ButcherTableau, EmbeddedTableau, EvolvedIncrement, ExplicitDaeFirstOrderRoot,
            ExplicitDaeZerothOrderRoot, Flat, IntegrableField, Product, StateEvolution, Unimodular,
            integrate_rkmk_dae_adaptive_first_order_root, rkmk_dae_step_first_order_root,
        },
        optimize::{EqualityConstraint, FirstOrderRootFinding, ZerothOrderRootFinding},
    },
    mechanics::{
        DeformationGradient, DeformationGradients, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, Times,
    },
    units::{Dissipation, Rate, Time},
};
use std::ops::{Add, Mul};

use crate::constitutive::solid::elastic_plastic::bcs;
pub use crate::constitutive::solid::elastic_plastic::{
    AppliedLoad, ElasticPlasticOrViscoplastic, PlasticTangents,
};

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

/// The internal state `(F_p, Y)` evolves as `F_p` on the unimodular group
/// (`Reference → Intermediate`, so its algebra element `D_p Δt` is
/// `Intermediate → Intermediate`) and the hardening variable `Y` additively. The
/// rate is `(D_p, Ẏ)`, from the model's [`plastic_evolution`] (`D_p` recovered as
/// `Ḟ_p F_p⁻¹`), driven by the total deformation gradient through the Mandel
/// stress. Blanket over any [`ElasticViscoplastic`] model — not
/// `Canonical`-specific — so a hybrid composition gets it automatically as
/// soon as it implements [`ElasticViscoplastic<Y>`].
///
/// [`plastic_evolution`]: Viscoplastic::plastic_evolution
impl<C, Y> StateEvolution<Time, Y> for C
where
    C: ElasticViscoplastic<Y>,
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

/// RKMK-DAE return-map methods for elastic-viscoplastic solid constitutive
/// models. The sibling of [`FirstOrderRoot`] that keeps `F_p` on its manifold
/// instead of marching it additively, by resolving `F` from equilibrium at
/// every stage abscissa rather than freezing it across the window — so the
/// coupling is the tableau's own order, not first order. Blanket over any
/// [`ElasticViscoplastic`] model, same as [`FirstOrderRoot`] itself.
pub trait RootRkmkDae<Y>
where
    Y: Differentiate + Tensor,
{
    /// `F` is re-solved from equilibrium at every stage abscissa of the
    /// window while `F_p` advances on its group, generic over the hardening
    /// variable `Y`. Stage `i` reconstructs `F_p` at `σᵢ`, solves
    /// `P(F, F_p^i) - λ(t + cᵢ Δt) - P_0 = 0` for `F` there, and evaluates the
    /// plastic rate at that consistent pair.
    ///
    /// This is the half-explicit RK treatment of the index-1 DAE that
    /// [`FirstOrderRoot::root`] already performs, with the state leg moved off
    /// the additive march onto `expm`/`dexpinv` — so `det F_p = 1` is kept
    /// rather than drifting.
    fn root_rkmk_dae<Tab: ButcherTableau>(
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
    >;
    /// As [`Self::root_rkmk_dae`], but the whole span is stepped under
    /// embedded (`Tab::D`) error control rather than on the supplied load
    /// grid.
    ///
    /// Two times in `applied_load` give only the span, and the controller's
    /// own accepted steps are reported. More than two are requested report
    /// times — the convention of the flat DAE loop — and `F_p` is served at
    /// each from the geodesic `HermiteSegment` of the accepted step
    /// containing it, so it is on the unimodular group at every reported time
    /// and not just at the accepted ones; `F` is then re-solved from
    /// equilibrium there.
    fn root_rkmk_dae_adaptive<Tab: EmbeddedTableau>(
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
    >;
}

impl<C, Y> RootRkmkDae<Y> for C
where
    C: ElasticViscoplastic<Y>
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
    fn root_rkmk_dae<Tab: ButcherTableau>(
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
    #[allow(clippy::type_complexity)]
    fn root_rkmk_dae_adaptive<Tab: EmbeddedTableau>(
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
