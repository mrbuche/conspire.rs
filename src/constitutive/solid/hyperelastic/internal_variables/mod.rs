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
        Quantity, Tensor, TensorArray, TensorTuple,
        optimize::{
            EqualityConstraint, FirstOrderOptimization, SecondOrderOptimizationBlock, SolveStrategy,
        },
    },
    mechanics::{
        DeformationGradient, FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness,
    },
    units::EnergyDensity,
};

/// Required methods for hyperelastic solid constitutive models with internal variables.
pub trait HyperelasticIV<V>
where
    Self: ElasticIV<V>,
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
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError>;
}

/// First-order minimization methods for hyperelastic solid constitutive models with internal variables.
pub trait FirstOrderMinimize<V> {
    /// Type representing all residuals.
    type Residuals;
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
        solver: impl FirstOrderOptimization<Quantity<EnergyDensity>, Self::Residuals, Self::Variables>,
    ) -> Result<(DeformationGradient, V), ConstitutiveError>;
}

/// Second-order minimization methods for hyperelastic solid constitutive models with internal variables.
pub trait SecondOrderMinimize<V>
where
    Self: ElasticIV<V>,
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
            Quantity<EnergyDensity>,
            DeformationGradient,
            V,
            FirstPiolaKirchhoffStress,
            <Self as ElasticIV<V>>::Residual,
            FirstPiolaKirchhoffTangentStiffness,
            Self::TangentVu,
            Self::TangentUv,
            Self::TangentVv,
        >,
        strategy: SolveStrategy,
    ) -> Result<(DeformationGradient, V), ConstitutiveError>;
}

impl<T, V> FirstOrderMinimize<V> for T
where
    T: HyperelasticIV<V>,
    T: ElasticIV<V>,
    V: Tensor,
{
    type Residuals = TensorTuple<FirstPiolaKirchhoffStress, <T as ElasticIV<V>>::Residual>;
    type Variables = TensorTuple<DeformationGradient, V>;
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderOptimization<Quantity<EnergyDensity>, Self::Residuals, Self::Variables>,
    ) -> Result<(DeformationGradient, V), ConstitutiveError> {
        let (matrix, vector) = bcs(self, applied_load);
        let solution = solver
            .minimize(
                |variables: &Self::Variables| {
                    let (deformation_gradient, internal_variables) = variables.into();
                    Ok(self
                        .helmholtz_free_energy_density(deformation_gradient, internal_variables)?)
                },
                |variables: &Self::Variables| {
                    let (deformation_gradient, internal_variables) = variables.into();
                    Ok(TensorTuple::from((
                        self.first_piola_kirchhoff_stress(
                            deformation_gradient,
                            internal_variables,
                        )?,
                        self.internal_variables_residual(deformation_gradient, internal_variables)?,
                    )))
                },
                Self::Variables::from((
                    DeformationGradient::identity(),
                    self.internal_variables_initial(),
                )),
                EqualityConstraint::Linear(matrix, vector),
            )
            .map_err(|error| ConstitutiveError::upstream(error, self))?;
        Ok(solution.into())
    }
}

impl<T, V> SecondOrderMinimize<V> for T
where
    T: HyperelasticIV<V>,
    V: Tensor,
{
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        solver: impl SecondOrderOptimizationBlock<
            Quantity<EnergyDensity>,
            DeformationGradient,
            V,
            FirstPiolaKirchhoffStress,
            <Self as ElasticIV<V>>::Residual,
            FirstPiolaKirchhoffTangentStiffness,
            Self::TangentVu,
            Self::TangentUv,
            Self::TangentVv,
        >,
        strategy: SolveStrategy,
    ) -> Result<(DeformationGradient, V), ConstitutiveError> {
        let (constraint_external, constraint_internal) = bcs_block(self, applied_load);
        solver
            .minimize_block(
                |deformation_gradient: &DeformationGradient, internal_variables: &V| {
                    Ok(self
                        .helmholtz_free_energy_density(deformation_gradient, internal_variables)?)
                },
                |deformation_gradient: &DeformationGradient, internal_variables: &V| {
                    Ok(self
                        .first_piola_kirchhoff_stress(deformation_gradient, internal_variables)?)
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
                None,
                strategy,
            )
            .map_err(|error| ConstitutiveError::upstream(error, self))
    }
}
