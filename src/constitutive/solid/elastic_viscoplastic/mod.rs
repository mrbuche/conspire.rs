//! Elastic-viscoplastic solid constitutive models.
//!
//! ---
//!
#![doc = include_str!("doc.md")]

#[cfg(feature = "doc")]
pub mod doc;

#[cfg(test)]
pub mod test;

mod almansi_hamel_eulerian;

pub use self::almansi_hamel_eulerian::AlmansiHamelEulerian;

use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::viscoplastic::{
            Viscoplastic, ViscoplasticEvolution, ViscoplasticEvolutionHistory,
            ViscoplasticStateVariables, ViscoplasticStateVariablesHistory,
        },
    },
    math::{
        ContractWith, Differentiate, Quantity, Rank2, Tensor, TensorArray, Vector,
        integrate::{ExplicitDaeFirstOrderRoot, ExplicitDaeZerothOrderRoot},
        optimize::{EqualityConstraint, FirstOrderRootFinding, ZerothOrderRootFinding},
    },
    mechanics::{
        DeformationGradient, DeformationGradients, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, Times,
    },
    units::{Dissipation, Time},
};

use crate::constitutive::solid::elastic_plastic::bcs;
pub use crate::constitutive::solid::elastic_plastic::{AppliedLoad, ElasticPlasticOrViscoplastic};

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
