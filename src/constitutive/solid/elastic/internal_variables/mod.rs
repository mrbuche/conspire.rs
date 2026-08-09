//! Elastic solid constitutive models with internal variables.

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, elastic::AppliedLoad},
    },
    math::{
        ContractFirstSecondWithSecond, ContractSecondWithFirst, IDENTITY, Matrix, Rank2, Tensor,
        TensorArray, TensorTuple, Vector,
        optimize::{
            EqualityConstraint, FirstOrderRootFindingBlock, SolveStrategy, ZerothOrderRootFinding,
        },
        sparse::CscMatrix,
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
            .contract_first_second_with_second(deformation_gradient, deformation_gradient)
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
            .contract_second_with_first(&deformation_gradient_inverse_transpose)
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
            .contract_first_second_with_second(
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
    /// Returns the indices of the internal variables held at zero.
    ///
    /// These fix any gauge freedom in how the internal variables are parameterized.
    fn internal_variables_fixed(&self) -> &[usize];
    /// Calculates and returns the tangents of the coupled system.
    fn tangents(
        &self,
        deformation_gradient: &DeformationGradient,
        internal_variables: &V,
    ) -> Result<(FirstPiolaKirchhoffTangentStiffness, T1, T2, T3), ConstitutiveError>;
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
    /// Solve for the unknown components of the deformation gradient under an applied load.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F}) - \boldsymbol{\lambda} - \mathbf{P}_0 = \mathbf{0}
    /// ```
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderRootFindingBlock<
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
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderRootFindingBlock<
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
        let (constraint_global, constraint_local) = bcs_block(self, applied_load);
        match solver.root_block(
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
            constraint_global,
            constraint_local,
            None,
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

#[doc(hidden)]
pub fn bcs_block<C, V, T1, T2, T3>(
    model: &C,
    applied_load: AppliedLoad,
) -> ((CscMatrix, Vector), (CscMatrix, Vector))
where
    C: ElasticIV<V, T1, T2, T3>,
    V: Tensor,
{
    let fixed = model.internal_variables_fixed();
    let num_internal_variables = model.internal_variables_initial().size();
    let pattern_vars = fixed.iter().enumerate().map(|(i, &j)| (i, j)).collect();
    let mut matrix_vars =
        CscMatrix::from_pattern(fixed.len(), num_internal_variables, pattern_vars);
    matrix_vars.fill(|_, _| 1.0);
    let local = (matrix_vars, Vector::zero(fixed.len()));
    let (vector, pattern) = match applied_load {
        AppliedLoad::UniaxialStress(deformation_gradient_11) => {
            let mut vector = Vector::zero(4);
            vector[0] = deformation_gradient_11;
            (vector, vec![(0, 0), (1, 1), (2, 2), (3, 5)])
        }
        AppliedLoad::BiaxialStress(deformation_gradient_11, deformation_gradient_22) => {
            let mut vector = Vector::zero(5);
            vector[0] = deformation_gradient_11;
            vector[4] = deformation_gradient_22;
            (vector, vec![(0, 0), (1, 1), (2, 2), (3, 5), (4, 4)])
        }
    };
    let mut matrix = CscMatrix::from_pattern(vector.len(), 9, pattern);
    matrix.fill(|_, _| 1.0);
    ((matrix, vector), local)
}

#[doc(hidden)]
pub fn bcs<C, V, T1, T2, T3>(model: &C, applied_load: AppliedLoad) -> (Matrix, Vector)
where
    C: ElasticIV<V, T1, T2, T3>,
    V: Tensor,
{
    let fixed = model.internal_variables_fixed();
    let num_internal_variables = model.internal_variables_initial().size();
    let num_deformation_gradient = 9;
    let (num_constraints, prescribed) = match applied_load {
        AppliedLoad::UniaxialStress(deformation_gradient_11) => (
            4,
            vec![
                (0, 0, deformation_gradient_11),
                (1, 1, 0.0),
                (2, 2, 0.0),
                (3, 5, 0.0),
            ],
        ),
        AppliedLoad::BiaxialStress(deformation_gradient_11, deformation_gradient_22) => (
            5,
            vec![
                (0, 0, deformation_gradient_11),
                (1, 1, 0.0),
                (2, 2, 0.0),
                (3, 5, 0.0),
                (4, 4, deformation_gradient_22),
            ],
        ),
    };
    let mut matrix = Matrix::zero(
        num_constraints + fixed.len(),
        num_deformation_gradient + num_internal_variables,
    );
    let mut vector = Vector::zero(num_constraints + fixed.len());
    prescribed.iter().for_each(|&(row, column, value)| {
        matrix[row][column] = 1.0;
        vector[row] = value
    });
    fixed
        .iter()
        .enumerate()
        .for_each(|(i, &j)| matrix[num_constraints + i][num_deformation_gradient + j] = 1.0);
    (matrix, vector)
}
