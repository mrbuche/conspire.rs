//! Hyperelastic-viscoplastic solid constitutive models.

#[cfg(test)]
pub mod test;

mod hencky;
mod saint_venant_kirchhoff;

pub use hencky::Hencky;
pub use saint_venant_kirchhoff::SaintVenantKirchhoff;

use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::viscoplastic::{ViscoplasticStateVariables, ViscoplasticStateVariablesHistory},
        solid::elastic_viscoplastic::{AppliedLoad, ElasticViscoplastic},
    },
    math::{
        Matrix, Tensor, TensorArray, Vector,
        integrate::{ExplicitDaeFirstOrderMinimize, ExplicitDaeSecondOrderMinimize},
        optimize::{EqualityConstraint, FirstOrderOptimization, SecondOrderOptimization},
    },
    mechanics::{
        DeformationGradient, DeformationGradientPlastic, DeformationGradients,
        FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness, Scalar, Times,
    },
};

/// Required methods for hyperelastic-viscoplastic solid constitutive models.
pub trait HyperelasticViscoplastic<Y>
where
    Self: ElasticViscoplastic<Y>,
    Y: Tensor,
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
    ) -> Result<Scalar, ConstitutiveError>;
}

/// First-order minimization methods for elastic-viscoplastic solid constitutive models.
pub trait FirstOrderMinimize<Y>
where
    Y: Tensor,
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
            Scalar,
            ViscoplasticStateVariables<Y>,
            DeformationGradient,
            ViscoplasticStateVariablesHistory<Y>,
            DeformationGradients,
        >,
        solver: impl FirstOrderOptimization<Scalar, DeformationGradient>,
    ) -> Result<
        (
            Times,
            DeformationGradients,
            ViscoplasticStateVariablesHistory<Y>,
        ),
        ConstitutiveError,
    >;
}

/// Second-order minimization methods for elastic-viscoplastic solid constitutive models.
pub trait SecondOrderMinimize<Y>
where
    Y: Tensor,
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
            Scalar,
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            ViscoplasticStateVariables<Y>,
            DeformationGradient,
            ViscoplasticStateVariablesHistory<Y>,
            DeformationGradients,
        >,
        solver: impl SecondOrderOptimization<
            Scalar,
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
    Y: Tensor,
{
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ExplicitDaeFirstOrderMinimize<
            Scalar,
            ViscoplasticStateVariables<Y>,
            DeformationGradient,
            ViscoplasticStateVariablesHistory<Y>,
            DeformationGradients,
        >,
        solver: impl FirstOrderOptimization<Scalar, DeformationGradient>,
    ) -> Result<
        (
            Times,
            DeformationGradients,
            ViscoplasticStateVariablesHistory<Y>,
        ),
        ConstitutiveError,
    > {
        match match applied_load {
            AppliedLoad::UniaxialStress(deformation_gradient_11, time) => {
                let mut matrix = Matrix::zero(4, 9);
                let mut vector = Vector::zero(4);
                matrix[0][0] = 1.0;
                matrix[1][1] = 1.0;
                matrix[2][2] = 1.0;
                matrix[3][5] = 1.0;
                integrator.integrate(
                    |_: Scalar,
                     state_variables: &ViscoplasticStateVariables<Y>,
                     deformation_gradient: &DeformationGradient| {
                        Ok(self.state_variables_evolution(deformation_gradient, state_variables)?)
                    },
                    |_: Scalar,
                     state_variables: &ViscoplasticStateVariables<Y>,
                     deformation_gradient: &DeformationGradient| {
                        let deformation_gradient_p = &state_variables.0;
                        Ok(self.helmholtz_free_energy_density(
                            deformation_gradient,
                            deformation_gradient_p,
                        )?)
                    },
                    |_: Scalar,
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
                    |t: Scalar| {
                        vector[0] = deformation_gradient_11(t);
                        EqualityConstraint::Linear(matrix.clone(), vector.clone())
                    },
                )
            }
        } {
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

impl<C, Y> SecondOrderMinimize<Y> for C
where
    C: HyperelasticViscoplastic<Y>,
    Y: Tensor,
{
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ExplicitDaeSecondOrderMinimize<
            Scalar,
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            ViscoplasticStateVariables<Y>,
            DeformationGradient,
            ViscoplasticStateVariablesHistory<Y>,
            DeformationGradients,
        >,
        solver: impl SecondOrderOptimization<
            Scalar,
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
        match match applied_load {
            AppliedLoad::UniaxialStress(deformation_gradient_11, time) => {
                let mut matrix = Matrix::zero(4, 9);
                let mut vector = Vector::zero(4);
                matrix[0][0] = 1.0;
                matrix[1][1] = 1.0;
                matrix[2][2] = 1.0;
                matrix[3][5] = 1.0;
                integrator.integrate(
                    |_: Scalar,
                     state_variables: &ViscoplasticStateVariables<Y>,
                     deformation_gradient: &DeformationGradient| {
                        Ok(self.state_variables_evolution(deformation_gradient, state_variables)?)
                    },
                    |_: Scalar,
                     state_variables: &ViscoplasticStateVariables<Y>,
                     deformation_gradient: &DeformationGradient| {
                        let deformation_gradient_p = &state_variables.0;
                        Ok(self.helmholtz_free_energy_density(
                            deformation_gradient,
                            deformation_gradient_p,
                        )?)
                    },
                    |_: Scalar,
                     state_variables: &ViscoplasticStateVariables<Y>,
                     deformation_gradient: &DeformationGradient| {
                        let deformation_gradient_p = &state_variables.0;
                        Ok(self.first_piola_kirchhoff_stress(
                            deformation_gradient,
                            deformation_gradient_p,
                        )?)
                    },
                    |_: Scalar,
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
                    |t: Scalar| {
                        vector[0] = deformation_gradient_11(t);
                        EqualityConstraint::Linear(matrix.clone(), vector.clone())
                    },
                    None,
                )
            }
        } {
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
