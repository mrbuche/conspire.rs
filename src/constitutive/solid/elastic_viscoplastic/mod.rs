//! Elastic-viscoplastic solid constitutive models.

use crate::math::unit::Time;
use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::viscoplastic::{
            Viscoplastic, ViscoplasticEvolution, ViscoplasticEvolutionHistory,
            ViscoplasticStateVariables, ViscoplasticStateVariablesHistory,
        },
    },
    math::{
        Differentiate, Quantity, Rank2, Tensor, TensorArray, Vector,
        integrate::{ExplicitDaeFirstOrderRoot, ExplicitDaeZerothOrderRoot},
        optimize::{EqualityConstraint, FirstOrderRootFinding, ZerothOrderRootFinding},
    },
    mechanics::{
        DeformationGradient, DeformationGradients, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, Times,
    },
};

use crate::constitutive::solid::elastic_plastic::bcs;
pub use crate::constitutive::solid::elastic_plastic::{AppliedLoad, ElasticPlasticOrViscoplastic};

/// Required methods for elastic-viscoplastic solid constitutive models.
pub trait ElasticViscoplastic<Y>
where
    Self: ElasticPlasticOrViscoplastic + Viscoplastic<Y>,
    Y: Differentiate + Tensor,
{
    /// Calculates and returns the evolution of the state variables.
    fn state_variables_evolution(
        &self,
        deformation_gradient: &DeformationGradient,
        state_variables: &ViscoplasticStateVariables<Y>,
    ) -> Result<ViscoplasticEvolution<Y>, ConstitutiveError> {
        let deformation_gradient_p = &state_variables.0;
        let jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let cauchy_stress = self.cauchy_stress(deformation_gradient, deformation_gradient_p)?;
        let mandel_stress = (deformation_gradient_e.transpose()
            * cauchy_stress
            * deformation_gradient_e.inverse_transpose())
            * jacobian;
        self.plastic_evolution(mandel_stress, state_variables)
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
        match integrator.integrate(
            |_: Quantity<Time>,
             state_variables: &ViscoplasticStateVariables<Y>,
             deformation_gradient: &DeformationGradient| {
                Ok(self.state_variables_evolution(deformation_gradient, state_variables)?)
            },
            |_: Quantity<Time>,
             state_variables: &ViscoplasticStateVariables<Y>,
             deformation_gradient: &DeformationGradient| {
                let deformation_gradient_p = &state_variables.0;
                Ok(self
                    .first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?)
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
        ) {
            Ok((times, state_variables, _, deformation_gradients)) => {
                Ok((times, deformation_gradients, state_variables))
            }
            Err(error) => Err(ConstitutiveError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
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
        match integrator.integrate(
            |_: Quantity<Time>,
             state_variables: &ViscoplasticStateVariables<Y>,
             deformation_gradient: &DeformationGradient| {
                Ok(self.state_variables_evolution(deformation_gradient, state_variables)?)
            },
            |_: Quantity<Time>,
             state_variables: &ViscoplasticStateVariables<Y>,
             deformation_gradient: &DeformationGradient| {
                let deformation_gradient_p = &state_variables.0;
                Ok(self
                    .first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?)
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
        ) {
            Ok((times, state_variables, _, deformation_gradients)) => {
                Ok((times, deformation_gradients, state_variables))
            }
            Err(error) => Err(ConstitutiveError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
