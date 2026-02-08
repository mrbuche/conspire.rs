//! Elastic-viscoplastic solid constitutive models.

use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::{
            plastic::{StateVariables, StateVariablesHistory},
            viscoplastic::Viscoplastic,
        },
    },
    math::{
        Matrix, Rank2, TensorArray, Vector,
        integrate::{ExplicitDaeFirstOrderRoot, ExplicitDaeZerothOrderRoot},
        optimize::{EqualityConstraint, FirstOrderRootFinding, ZerothOrderRootFinding},
    },
    mechanics::{
        DeformationGradient, DeformationGradientPlastic, DeformationGradients,
        FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness, Scalar, Times,
    },
};

pub use crate::constitutive::solid::elastic_plastic::{AppliedLoad, ElasticPlasticOrViscoplastic};

/// Required methods for elastic-viscoplastic solid constitutive models.
pub trait ElasticViscoplastic
where
    Self: ElasticPlasticOrViscoplastic + Viscoplastic,
{
    /// Calculates and returns the evolution of the state variables.
    fn state_variables_evolution(
        &self,
        deformation_gradient: &DeformationGradient,
        state_variables: &StateVariables,
    ) -> Result<StateVariables, ConstitutiveError> {
        let (deformation_gradient_p, &equivalent_plastic_strain) = state_variables.into();
        let jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let cauchy_stress = self.cauchy_stress(deformation_gradient, deformation_gradient_p)?;
        let mandel_stress_e = (deformation_gradient_e.transpose()
            * cauchy_stress
            * deformation_gradient_e.inverse_transpose())
            * jacobian;
        self.plastic_evolution(
            mandel_stress_e,
            deformation_gradient_p,
            equivalent_plastic_strain,
        )
    }
}

/// Zeroth-order root-finding methods for elastic-viscoplastic solid constitutive models.
pub trait ZerothOrderRoot {
    /// Solve for the unknown components of the deformation gradients under an applied load.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F},\mathbf{F}_\mathrm{p}) - \boldsymbol{\lambda} - \mathbf{P}_0 = \mathbf{0}
    /// ```
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ExplicitDaeZerothOrderRoot<
            StateVariables,
            DeformationGradient,
            StateVariablesHistory,
            DeformationGradients,
        >,
        solver: impl ZerothOrderRootFinding<DeformationGradient>,
    ) -> Result<(Times, DeformationGradients, StateVariablesHistory), ConstitutiveError>;
}

/// First-order root-finding methods for elastic-viscoplastic solid constitutive models.
pub trait FirstOrderRoot {
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
            StateVariables,
            DeformationGradient,
            StateVariablesHistory,
            DeformationGradients,
        >,
        solver: impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
    ) -> Result<(Times, DeformationGradients, StateVariablesHistory), ConstitutiveError>;
}

impl<T> ZerothOrderRoot for T
where
    T: ElasticViscoplastic,
{
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ExplicitDaeZerothOrderRoot<
            StateVariables,
            DeformationGradient,
            StateVariablesHistory,
            DeformationGradients,
        >,
        solver: impl ZerothOrderRootFinding<DeformationGradient>,
    ) -> Result<(Times, DeformationGradients, StateVariablesHistory), ConstitutiveError> {
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
                     state_variables: &StateVariables,
                     deformation_gradient: &DeformationGradient| {
                        Ok(self.state_variables_evolution(deformation_gradient, state_variables)?)
                    },
                    |_: Scalar,
                     state_variables: &StateVariables,
                     deformation_gradient: &DeformationGradient| {
                        let (deformation_gradient_p, _) = state_variables.into();
                        Ok(self.first_piola_kirchhoff_stress(
                            deformation_gradient,
                            deformation_gradient_p,
                        )?)
                    },
                    solver,
                    time,
                    (
                        StateVariables::from((DeformationGradientPlastic::identity(), 0.0)),
                        DeformationGradient::identity(),
                    ),
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

impl<T> FirstOrderRoot for T
where
    T: ElasticViscoplastic,
{
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ExplicitDaeFirstOrderRoot<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            StateVariables,
            DeformationGradient,
            StateVariablesHistory,
            DeformationGradients,
        >,
        solver: impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
    ) -> Result<(Times, DeformationGradients, StateVariablesHistory), ConstitutiveError> {
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
                     state_variables: &StateVariables,
                     deformation_gradient: &DeformationGradient| {
                        Ok(self.state_variables_evolution(deformation_gradient, state_variables)?)
                    },
                    |_: Scalar,
                     state_variables: &StateVariables,
                     deformation_gradient: &DeformationGradient| {
                        let (deformation_gradient_p, _) = state_variables.into();
                        Ok(self.first_piola_kirchhoff_stress(
                            deformation_gradient,
                            deformation_gradient_p,
                        )?)
                    },
                    |_: Scalar,
                     state_variables: &StateVariables,
                     deformation_gradient: &DeformationGradient| {
                        let (deformation_gradient_p, _) = state_variables.into();
                        Ok(self.first_piola_kirchhoff_tangent_stiffness(
                            deformation_gradient,
                            deformation_gradient_p,
                        )?)
                    },
                    solver,
                    time,
                    (
                        StateVariables::from((DeformationGradientPlastic::identity(), 0.0)),
                        DeformationGradient::identity(),
                    ),
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
