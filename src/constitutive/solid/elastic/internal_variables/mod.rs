//! Elastic solid constitutive models with internal variables.

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, elastic::AppliedLoad},
    },
    math::{
        ContractFirstSecondIndicesWithSecondIndicesOf, ContractSecondIndexWithFirstIndexOf,
        IDENTITY, Matrix, Rank2, Tensor, TensorArray, TensorTuple, Vector,
        optimize::{EqualityConstraint, FirstOrderRootFinding, ZerothOrderRootFinding},
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, DeformationGradient, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, SecondPiolaKirchhoffStress,
        SecondPiolaKirchhoffTangentStiffness,
    },
};

/// Required methods for elastic solid constitutive models with internal variables.
pub trait ElasticIV<V, T1, T2, T3>
where
    Self: Solid,
{
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma} = J^{-1}\mathbf{P}\cdot\mathbf{F}^T
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        internal_variables: &V,
    ) -> Result<CauchyStress, ConstitutiveError> {
        Ok(deformation_gradient
            * self.second_piola_kirchhoff_stress(deformation_gradient, internal_variables)?
            * deformation_gradient.transpose()
            / deformation_gradient.determinant())
    }
    /// Calculates and returns the tangent stiffness associated with the Cauchy stress.
    ///
    /// ```math
    /// \mathcal{T}_{ijkL} = \frac{\partial\sigma_{ij}}{\partial F_{kL}} = J^{-1} \mathcal{G}_{MNkL} F_{iM} F_{jN} - \sigma_{ij} F_{kL}^{-T} + \left(\delta_{jk}\sigma_{is} + \delta_{ik}\sigma_{js}\right)F_{sL}^{-T}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        internal_variables: &V,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let cauchy_stress = self.cauchy_stress(deformation_gradient, internal_variables)?;
        let some_stress = &cauchy_stress * &deformation_gradient_inverse_transpose;
        Ok(self
            .second_piola_kirchhoff_tangent_stiffness(deformation_gradient, internal_variables)?
            .contract_first_second_indices_with_second_indices_of(
                deformation_gradient,
                deformation_gradient,
            )
            / deformation_gradient.determinant()
            - CauchyTangentStiffness::dyad_ij_kl(
                &cauchy_stress,
                &deformation_gradient_inverse_transpose,
            )
            + CauchyTangentStiffness::dyad_il_kj(&some_stress, &IDENTITY)
            + CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, &some_stress))
    }
    /// Calculates and returns the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{P} = J\boldsymbol{\sigma}\cdot\mathbf{F}^{-T}
    /// ```
    fn first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        internal_variables: &V,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        Ok(
            self.cauchy_stress(deformation_gradient, internal_variables)?
                * deformation_gradient.inverse_transpose()
                * deformation_gradient.determinant(),
        )
    }
    /// Calculates and returns the tangent stiffness associated with the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{C}_{iJkL} = \frac{\partial P_{iJ}}{\partial F_{kL}} = J \mathcal{T}_{iskL} F_{sJ}^{-T} + P_{iJ} F_{kL}^{-T} - P_{iL} F_{kJ}^{-T}
    /// ```
    fn first_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        internal_variables: &V,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let first_piola_kirchhoff_stress =
            self.first_piola_kirchhoff_stress(deformation_gradient, internal_variables)?;
        Ok(self
            .cauchy_tangent_stiffness(deformation_gradient, internal_variables)?
            .contract_second_index_with_first_index_of(&deformation_gradient_inverse_transpose)
            * deformation_gradient.determinant()
            + FirstPiolaKirchhoffTangentStiffness::dyad_ij_kl(
                &first_piola_kirchhoff_stress,
                &deformation_gradient_inverse_transpose,
            )
            - FirstPiolaKirchhoffTangentStiffness::dyad_il_kj(
                &first_piola_kirchhoff_stress,
                &deformation_gradient_inverse_transpose,
            ))
    }
    /// Calculates and returns the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{S} = \mathbf{F}^{-1}\cdot\mathbf{P}
    /// ```
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        internal_variables: &V,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        Ok(deformation_gradient.inverse()
            * self.first_piola_kirchhoff_stress(deformation_gradient, internal_variables)?)
    }
    /// Calculates and returns the tangent stiffness associated with the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{G}_{IJkL} = \frac{\partial S_{IJ}}{\partial F_{kL}} = \mathcal{C}_{mJkL}F_{mI}^{-T} - S_{LJ}F_{kI}^{-T} = J \mathcal{T}_{mnkL} F_{mI}^{-T} F_{nJ}^{-T} + S_{IJ} F_{kL}^{-T} - S_{IL} F_{kJ}^{-T} -S_{LJ} F_{kI}^{-T}
    /// ```
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        internal_variables: &V,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let deformation_gradient_inverse = deformation_gradient_inverse_transpose.transpose();
        let second_piola_kirchhoff_stress =
            self.second_piola_kirchhoff_stress(deformation_gradient, internal_variables)?;
        Ok(self
            .cauchy_tangent_stiffness(deformation_gradient, internal_variables)?
            .contract_first_second_indices_with_second_indices_of(
                &deformation_gradient_inverse,
                &deformation_gradient_inverse,
            )
            * deformation_gradient.determinant()
            + SecondPiolaKirchhoffTangentStiffness::dyad_ij_kl(
                &second_piola_kirchhoff_stress,
                &deformation_gradient_inverse_transpose,
            )
            - SecondPiolaKirchhoffTangentStiffness::dyad_il_kj(
                &second_piola_kirchhoff_stress,
                &deformation_gradient_inverse_transpose,
            )
            - SecondPiolaKirchhoffTangentStiffness::dyad_ik_jl(
                &deformation_gradient_inverse,
                &second_piola_kirchhoff_stress,
            ))
    }
    /// Returns the initial value for the internal variables.
    fn internal_variables_initial(&self) -> V;
    /// Calculates and returns the residual associated with the internal variables.
    fn internal_variables_residual(
        &self,
        deformation_gradient: &DeformationGradient,
        internal_variables: &V,
    ) -> Result<V, ConstitutiveError>;
    /// Calculates and returns the tangents associated with the internal variables.
    fn internal_variables_tangents(
        &self,
        deformation_gradient: &DeformationGradient,
        internal_variables: &V,
    ) -> Result<(T1, T2, T3), ConstitutiveError>;
    /// Returns the constraint indices for the internal variables.
    fn internal_variables_constraints(&self) -> (&[usize], usize);
}

/// Zeroth-order root-finding methods for elastic solid constitutive models with internal variables.
pub trait ZerothOrderRoot<V, T1, T2, T3>
where
    V: Tensor,
{
    /// Type representing all variables.
    type Variables;
    /// Solve for the unknown components of the deformation gradient under an applied load.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F}) - \boldsymbol{\lambda} - \mathbf{P}_0 = \mathbf{0}
    /// ```
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl ZerothOrderRootFinding<Self::Variables>,
    ) -> Result<(DeformationGradient, V), ConstitutiveError>;
}

/// First-order root-finding methods for elastic solid constitutive models with internal variables.
pub trait FirstOrderRoot<V, T1, T2, T3>
where
    T1: Tensor,
    T2: Tensor,
    T3: Tensor,
    V: Tensor,
{
    /// Type representing all variables.
    type Variables;
    /// Solve for the unknown components of the deformation gradient under an applied load.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F}) - \boldsymbol{\lambda} - \mathbf{P}_0 = \mathbf{0}
    /// ```
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderRootFinding<
            Self::Variables,
            TensorTuple<FirstPiolaKirchhoffTangentStiffness, TensorTuple<T1, TensorTuple<T2, T3>>>,
            Self::Variables,
        >,
    ) -> Result<(DeformationGradient, V), ConstitutiveError>;
}

impl<T, V, T1, T2, T3> ZerothOrderRoot<V, T1, T2, T3> for T
where
    T: ElasticIV<V, T1, T2, T3>,
    V: Tensor,
{
    type Variables = TensorTuple<DeformationGradient, V>;
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl ZerothOrderRootFinding<Self::Variables>,
    ) -> Result<(DeformationGradient, V), ConstitutiveError> {
        let (matrix, vector) = bcs(self, applied_load);
        match solver.root(
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

impl<T, V, T1, T2, T3> FirstOrderRoot<V, T1, T2, T3> for T
where
    T1: Tensor,
    T2: Tensor,
    T3: Tensor,
    T: ElasticIV<V, T1, T2, T3>,
    V: Tensor,
{
    type Variables = TensorTuple<DeformationGradient, V>;
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderRootFinding<
            Self::Variables,
            TensorTuple<FirstPiolaKirchhoffTangentStiffness, TensorTuple<T1, TensorTuple<T2, T3>>>,
            Self::Variables,
        >,
    ) -> Result<(DeformationGradient, V), ConstitutiveError> {
        let (matrix, vector) = bcs(self, applied_load);
        match solver.root(
            |variables: &Self::Variables| {
                let (deformation_gradient, internal_variables) = variables.into();
                Ok(TensorTuple::from((
                    self.first_piola_kirchhoff_stress(deformation_gradient, internal_variables)?,
                    self.internal_variables_residual(deformation_gradient, internal_variables)?,
                )))
            },
            |variables: &Self::Variables| {
                let (deformation_gradient, internal_variables) = variables.into();
                let tangent_0 = self.first_piola_kirchhoff_tangent_stiffness(
                    deformation_gradient,
                    internal_variables,
                )?;
                let (tangent_1, tangent_2, tangent_3) =
                    self.internal_variables_tangents(deformation_gradient, internal_variables)?;
                Ok((tangent_0, (tangent_1, (tangent_2, tangent_3).into()).into()).into())
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

#[doc(hidden)]
pub fn bcs<C, V, T1, T2, T3>(model: &C, applied_load: AppliedLoad) -> (Matrix, Vector)
where
    C: ElasticIV<V, T1, T2, T3>,
{
    let (extra_constraints, num_vars) = model.internal_variables_constraints();
    match applied_load {
        AppliedLoad::UniaxialStress(deformation_gradient_11) => {
            let num_constraints = 4;
            let num_constraints_vars = extra_constraints.len();
            let mut matrix = Matrix::zero(num_constraints + num_constraints_vars, 9 + num_vars);
            let mut vector = Vector::zero(num_constraints + num_constraints_vars);
            matrix[0][0] = 1.0;
            matrix[1][1] = 1.0;
            matrix[2][2] = 1.0;
            matrix[3][5] = 1.0;
            let mut i = num_constraints;
            extra_constraints.iter().for_each(|&j| {
                matrix[i][j] = 1.0;
                i += 1;
            });
            vector[0] = deformation_gradient_11;
            (matrix, vector)
        }
        AppliedLoad::BiaxialStress(_, _) => {
            todo!()
        }
    }
}
