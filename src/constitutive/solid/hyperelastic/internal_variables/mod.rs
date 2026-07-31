//! Hyperelastic solid constitutive models with internal variables.

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::elastic::{
            AppliedLoad,
            internal_variables::{ElasticIV, bcs, bcs_block},
        },
    },
    math::{
        Scalar, Tensor, TensorArray, TensorTuple,
        optimize::{
            EqualityConstraint, FirstOrderOptimization, SecondOrderOptimizationBlock, SolveStrategy,
        },
    },
    mechanics::{
        DeformationGradient, FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness,
    },
};

/// Required methods for hyperelastic solid constitutive models with internal variables.
pub trait HyperelasticIV<V, T1, T2, T3>
where
    Self: ElasticIV<V, T1, T2, T3>,
{
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a = a(\mathbf{F})
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        internal_variables: &V,
    ) -> Result<Scalar, ConstitutiveError>;
}

/// First-order minimization methods for hyperelastic solid constitutive models with internal variables.
pub trait FirstOrderMinimize<V, T1, T2, T3> {
    /// Type representing all variables.
    type Variables;
    /// Solve for the unknown components of the deformation gradient under an applied load.
    ///
    /// ```math
    /// \Pi(\mathbf{F},\boldsymbol{\lambda}) = a(\mathbf{F}) - \boldsymbol{\lambda}:(\mathbf{F} - \mathbf{F}_0) - \mathbf{P}_0:\mathbf{F}
    /// ```
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderOptimization<Scalar, Self::Variables>,
    ) -> Result<(DeformationGradient, V), ConstitutiveError>;
}

/// Second-order minimization methods for hyperelastic solid constitutive models with internal variables.
pub trait SecondOrderMinimize<V, T1, T2, T3>
where
    T1: Tensor,
    T2: Tensor,
    T3: Tensor,
    V: Tensor,
{
    /// Solve for the unknown components of the deformation gradient under an applied load.
    ///
    /// ```math
    /// \Pi(\mathbf{F},\boldsymbol{\lambda}) = a(\mathbf{F}) - \boldsymbol{\lambda}:(\mathbf{F} - \mathbf{F}_0) - \mathbf{P}_0:\mathbf{F}
    /// ```
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        solver: impl SecondOrderOptimizationBlock<
            DeformationGradient,
            V,
            FirstPiolaKirchhoffStress,
            V,
            FirstPiolaKirchhoffTangentStiffness,
            T1,
            T2,
            T3,
        >,
        strategy: SolveStrategy,
    ) -> Result<(DeformationGradient, V), ConstitutiveError>;
}

impl<T, V, T1, T2, T3> FirstOrderMinimize<V, T1, T2, T3> for T
where
    T: HyperelasticIV<V, T1, T2, T3>,
    T1: Tensor,
    T2: Tensor,
    T3: Tensor,
    T: ElasticIV<V, T1, T2, T3>,
    V: Tensor,
{
    type Variables = TensorTuple<DeformationGradient, V>;
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderOptimization<Scalar, Self::Variables>,
    ) -> Result<(DeformationGradient, V), ConstitutiveError> {
        let (matrix, vector) = bcs(self, applied_load);
        match solver.minimize(
            |variables: &Self::Variables| {
                let (deformation_gradient, internal_variables) = variables.into();
                Ok(self.helmholtz_free_energy_density(deformation_gradient, internal_variables)?)
            },
            |variables: &Self::Variables| {
                let (deformation_gradient, internal_variables) = variables.into();
                Ok(TensorTuple::from((
                    self.first_piola_kirchhoff_stress(deformation_gradient, internal_variables)?,
                    self.internal_variables_residual(deformation_gradient, internal_variables)?,
                )))
            },
            Self::Variables::from((
                DeformationGradient::identity(),
                self.internal_variables_initial(),
            )),
            EqualityConstraint::Linear(matrix, vector),
        ) {
            Ok(solution) => Ok(solution.into()),
            Err(error) => Err(ConstitutiveError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<T, V, T1, T2, T3> SecondOrderMinimize<V, T1, T2, T3> for T
where
    T1: Tensor,
    T2: Tensor,
    T3: Tensor,
    T: HyperelasticIV<V, T1, T2, T3>,
    V: Tensor,
{
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        solver: impl SecondOrderOptimizationBlock<
            DeformationGradient,
            V,
            FirstPiolaKirchhoffStress,
            V,
            FirstPiolaKirchhoffTangentStiffness,
            T1,
            T2,
            T3,
        >,
        strategy: SolveStrategy,
    ) -> Result<(DeformationGradient, V), ConstitutiveError> {
        let (constraint_external, constraint_internal) = bcs_block(self, applied_load);
        match solver.minimize_block(
            |deformation_gradient: &DeformationGradient, internal_variables: &V| {
                Ok(self.helmholtz_free_energy_density(deformation_gradient, internal_variables)?)
            },
            |deformation_gradient: &DeformationGradient, internal_variables: &V| {
                Ok(self.first_piola_kirchhoff_stress(deformation_gradient, internal_variables)?)
            },
            |deformation_gradient: &DeformationGradient, internal_variables: &V| {
                Ok(self.internal_variables_residual(deformation_gradient, internal_variables)?)
            },
            |deformation_gradient: &DeformationGradient, internal_variables: &V| {
                Ok(self.tangents(deformation_gradient, internal_variables)?)
            },
            (
                DeformationGradient::identity(),
                self.internal_variables_initial(),
            ),
            constraint_external,
            constraint_internal,
            strategy,
        ) {
            Ok(solution) => Ok(solution),
            Err(error) => Err(ConstitutiveError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
