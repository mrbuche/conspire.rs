//! Elastic solid constitutive models with internal variables.

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, elastic::AppliedLoad},
    },
    math::{
        ContractFirstSecondWithSecond, ContractSecondWithFirst, Hessian, HessianBlock, IDENTITY,
        Jacobian, Matrix, Rank2, Tensor, TensorArray, TensorTuple, Vector,
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

/// The tangents of the coupled system, in the order the block solver takes them.
pub type Tangents<C, V> = (
    FirstPiolaKirchhoffTangentStiffness,
    <C as ElasticIV<V>>::TangentVu,
    <C as ElasticIV<V>>::TangentUv,
    <C as ElasticIV<V>>::TangentVv,
);

/// Required methods for elastic solid constitutive models with internal variables.
pub trait ElasticIV<V>
where
    Self: Solid,
{
    /// The residual associated with the internal variables.
    /// the internal variables are in equilibrium over, typically a stress.
    type Residual: Jacobian;
    /// The tangent of the internal variables residual with the deformation gradient.
    type TangentVu: HessianBlock;
    /// The tangent of the deformation gradient residual with the internal variables.
    type TangentUv: HessianBlock;
    /// The tangent of the internal variables residual with the internal variables.
    type TangentVv: Hessian + HessianBlock;
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
    ) -> Result<Self::Residual, ConstitutiveError>;
    /// Returns the indices of the internal variables held at zero.
    fn internal_variables_fixed(&self) -> &[usize];
    /// Calculates and returns the tangents of the coupled system.
    fn tangents(
        &self,
        deformation_gradient: &DeformationGradient,
        internal_variables: &V,
    ) -> Result<Tangents<Self, V>, ConstitutiveError>;
}

/// Zeroth-order root-finding methods for elastic solid constitutive models with internal variables.
pub trait ZerothOrderRoot<V>
where
    V: Tensor,
{
    /// Type representing all residuals.
    type Residuals;
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
        solver: impl ZerothOrderRootFinding<Self::Residuals, Self::Variables>,
    ) -> Result<(DeformationGradient, V), ConstitutiveError>;
}

/// First-order root-finding methods for elastic solid constitutive models with internal variables.
pub trait FirstOrderRoot<V>
where
    Self: ElasticIV<V>,
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
            <Self as ElasticIV<V>>::Residual,
            FirstPiolaKirchhoffTangentStiffness,
            Self::TangentVu,
            Self::TangentUv,
            Self::TangentVv,
        >,
        strategy: SolveStrategy,
    ) -> Result<(DeformationGradient, V), ConstitutiveError>;
}

impl<T, V> ZerothOrderRoot<V> for T
where
    T: ElasticIV<V>,
    V: Tensor,
{
    type Residuals = TensorTuple<FirstPiolaKirchhoffStress, <T as ElasticIV<V>>::Residual>;
    type Variables = TensorTuple<DeformationGradient, V>;
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl ZerothOrderRootFinding<Self::Residuals, Self::Variables>,
    ) -> Result<(DeformationGradient, V), ConstitutiveError> {
        let (matrix, vector) = bcs(self, applied_load);
        let solution = solver
            .root(
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

impl<T, V> FirstOrderRoot<V> for T
where
    T: ElasticIV<V>,
    V: Tensor,
{
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderRootFindingBlock<
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
        let (constraint_global, constraint_local) = bcs_block(self, applied_load);
        solver
            .root_block(
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
                constraint_global,
                constraint_local,
                None,
                strategy,
            )
            .map_err(|error| ConstitutiveError::upstream(error, self))
    }
}

#[doc(hidden)]
pub fn bcs_block<C, V>(
    model: &C,
    applied_load: AppliedLoad,
) -> ((CscMatrix, Vector), (CscMatrix, Vector))
where
    C: ElasticIV<V>,
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
pub fn bcs<C, V>(model: &C, applied_load: AppliedLoad) -> (Matrix, Vector)
where
    C: ElasticIV<V>,
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
