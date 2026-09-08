//! Elastic-plastic solid constitutive models.

mod canonical;

use crate::{
    EPSILON,
    constitutive::{
        ConstitutiveError,
        fluid::plastic::{
            Plastic, PlasticStateVariables, PlasticStateVariablesHistory, RateIndependentPlastic,
        },
        solid::Solid,
    },
    math::{
        ContractFirstSecondWithSecond, ContractSecondWithFirst, Current, IDENTITY, Intermediate,
        Matrix, Quantity, Rank2, Reference, TensorArray, TensorRank4, Vector,
        assert::perturbation,
        optimize::{
            EqualityConstraint, FirstOrderRootFinding, FirstOrderRootFindingBlock, SolveStrategy,
            ZerothOrderRootFinding,
        },
        sparse::CscMatrix,
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, DeformationGradient, DeformationGradientPlastic,
        DeformationGradients, FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness,
        MandelStressElastic, Scalar, SecondPiolaKirchhoffStress,
        SecondPiolaKirchhoffTangentStiffness, Times,
    },
    units::{Dimensionless, Stress, Time},
};

/// Possible applied loads.
pub enum AppliedLoad<'a> {
    /// Uniaxial stress given $`F_{11}`$.
    UniaxialStress(fn(Quantity<Time>) -> Scalar, &'a [Quantity<Time>]),
    /// Biaxial stress given $`F_{11}`$ and $`F_{22}`$.
    BiaxialStress(
        fn(Quantity<Time>) -> Scalar,
        fn(Quantity<Time>) -> Scalar,
        &'a [Quantity<Time>],
    ),
}

type Prescribed = Vec<(usize, fn(Quantity<Time>) -> Scalar)>;

#[doc(hidden)]
pub fn bcs(applied_load: AppliedLoad<'_>) -> (Matrix, Prescribed, &'_ [Quantity<Time>]) {
    let (mut matrix, prescribed, time) = match applied_load {
        AppliedLoad::UniaxialStress(deformation_gradient_11, time) => {
            (Matrix::zero(4, 9), vec![(0, deformation_gradient_11)], time)
        }
        AppliedLoad::BiaxialStress(deformation_gradient_11, deformation_gradient_22, time) => (
            Matrix::zero(5, 9),
            vec![(0, deformation_gradient_11), (4, deformation_gradient_22)],
            time,
        ),
    };
    matrix[0][0] = 1.0;
    matrix[1][1] = 1.0;
    matrix[2][2] = 1.0;
    matrix[3][5] = 1.0;
    if matrix.len() == 5 {
        matrix[4][4] = 1.0
    }
    (matrix, prescribed, time)
}

/// Required methods for elastic-plastic or elastic-viscoplastic solid constitutive models.
pub trait ElasticPlasticOrViscoplastic
where
    Self: Solid + Plastic,
{
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma} = \boldsymbol{\sigma}_\mathrm{e}
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyStress, ConstitutiveError> {
        Ok(deformation_gradient
            * self.second_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?
            * deformation_gradient.transpose()
            / deformation_gradient.determinant())
    }
    /// Calculates and returns the tangent stiffness associated with the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\mathcal{T}} = \boldsymbol{\mathcal{T}}_\mathrm{e}\cdot\mathbf{F}_\mathrm{p}^{-T}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let cauchy_stress = self.cauchy_stress(deformation_gradient, deformation_gradient_p)?;
        let some_stress = &cauchy_stress * &deformation_gradient_inverse_transpose;
        Ok(self
            .second_piola_kirchhoff_tangent_stiffness(deformation_gradient, deformation_gradient_p)?
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
    /// \mathbf{P} = \mathbf{P}_\mathrm{e}\cdot\mathbf{F}_\mathrm{p}^{-T}
    /// ```
    fn first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        Ok(
            self.cauchy_stress(deformation_gradient, deformation_gradient_p)?
                * deformation_gradient.inverse_transpose()
                * deformation_gradient.determinant(),
        )
    }
    /// Calculates and returns the tangent stiffness associated with the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{C}_{iJkL} = \mathcal{C}^\mathrm{e}_{iMkN} F_{MJ}^{\mathrm{p}-T} F_{NL}^{\mathrm{p}-T}
    /// ```
    fn first_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let first_piola_kirchhoff_stress =
            self.first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?;
        Ok(self
            .cauchy_tangent_stiffness(deformation_gradient, deformation_gradient_p)?
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
    /// Calculates and returns the Mandel stress.
    ///
    /// ```math
    /// \mathbf{M}_\mathrm{e} = J\mathbf{F}_\mathrm{e}^T\cdot\boldsymbol{\sigma}\cdot\mathbf{F}_\mathrm{e}^{-T}
    /// ```
    fn mandel_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<MandelStressElastic, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let cauchy_stress = self.cauchy_stress(deformation_gradient, deformation_gradient_p)?;
        Ok((deformation_gradient_e.transpose()
            * cauchy_stress
            * deformation_gradient_e.inverse_transpose())
            * jacobian)
    }
    /// Calculates and returns the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{S} = \mathbf{F}_\mathrm{p}^{-1}\cdot\mathbf{S}_\mathrm{e}\cdot\mathbf{F}_\mathrm{p}^{-T}
    /// ```
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        Ok(deformation_gradient.inverse()
            * self.first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?)
    }
    /// Calculates and returns the tangent stiffness associated with the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{G}_{IJkL} = \mathcal{G}^\mathrm{e}_{MNkO} F_{MI}^{\mathrm{p}-T} F_{NJ}^{\mathrm{p}-T} F_{OL}^{\mathrm{p}-T}
    /// ```
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let deformation_gradient_inverse = deformation_gradient_inverse_transpose.transpose();
        let second_piola_kirchhoff_stress =
            self.second_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?;
        Ok(self
            .cauchy_tangent_stiffness(deformation_gradient, deformation_gradient_p)?
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
}

/// Required methods for elastic-plastic solid constitutive models.
pub trait ElasticPlastic
where
    Self: ElasticPlasticOrViscoplastic + RateIndependentPlastic,
{
    /// Return mapping over one load step.
    ///
    /// Given the total deformation gradient and the previously converged plastic
    /// state, solves for the updated plastic state enforcing the Karush-Kuhn-Tucker
    /// conditions
    /// ```math
    /// f \leq 0, \qquad \Delta\gamma \geq 0, \qquad \Delta\gamma\, f = 0
    /// ```
    /// through an elastic predictor and, if the trial state lies outside the yield
    /// surface, a plastic corrector for the incremental multiplier $`\Delta\gamma`$.
    /// The flow direction is frozen at the trial state and the plastic deformation
    /// gradient is updated by the exponential map
    /// ```math
    /// \mathbf{F}_\mathrm{p}^{n+1} = \exp(\Delta\gamma\,\mathbf{N})\cdot\mathbf{F}_\mathrm{p}^{n},
    /// ```
    /// which is unimodular for the trace-free $`\mathbf{N}`$ and so needs no step
    /// limit. The scalar consistency equation is solved by a bracketed Newton
    /// iteration that falls back to bisection.
    fn return_map(
        &self,
        deformation_gradient: &DeformationGradient,
        state_variables: &PlasticStateVariables,
    ) -> Result<PlasticStateVariables, ConstitutiveError> {
        let (deformation_gradient_p, &equivalent_plastic_strain): (
            &DeformationGradientPlastic,
            &Quantity,
        ) = state_variables.into();
        let deviatoric_trial = self
            .mandel_stress(deformation_gradient, deformation_gradient_p)?
            .deviatoric();
        if self
            .yield_function(&deviatoric_trial, equivalent_plastic_strain)?
            .value()
            <= 0.0
        {
            return Ok(state_variables.clone());
        }
        let flow_direction = {
            let direction = self.flow_direction(&deviatoric_trial)?;
            (&direction + direction.transpose()) * 0.5
        };
        let plastic_deformation_gradient =
            |plastic_multiplier: Scalar| -> Result<DeformationGradientPlastic, ConstitutiveError> {
                Ok((&flow_direction * plastic_multiplier)
                    .expm()
                    .map_err(|error| ConstitutiveError::custom(format!("{error:?}"), self))?
                    * deformation_gradient_p)
            };
        let residual = |plastic_multiplier: Scalar| -> Result<Scalar, ConstitutiveError> {
            let deviatoric = self
                .mandel_stress(
                    deformation_gradient,
                    &plastic_deformation_gradient(plastic_multiplier)?,
                )?
                .deviatoric();
            Ok(self
                .yield_function(
                    &deviatoric,
                    equivalent_plastic_strain + Quantity::new(plastic_multiplier),
                )?
                .value())
        };
        let (mut lower, mut upper) = (0.0, 1e-3);
        while residual(upper)? > 0.0 {
            upper *= 2.0;
            if upper > 16.0 {
                return Err(ConstitutiveError::custom(
                    "Return mapping failed to bracket the plastic multiplier.",
                    self,
                ));
            }
        }
        let tolerance = 1e-13 * self.initial_yield_stress().value().max(1.0);
        let mut plastic_multiplier = 0.5 * (lower + upper);
        for _ in 0..40 {
            let value = residual(plastic_multiplier)?;
            if value.abs() <= tolerance || upper - lower <= 1e-15 * (1.0 + plastic_multiplier) {
                break;
            }
            if value > 0.0 {
                lower = plastic_multiplier;
            } else {
                upper = plastic_multiplier;
            }
            let step = EPSILON * plastic_multiplier.max(1e-3);
            let slope = (residual(plastic_multiplier + step)? - value) / step;
            let newton = plastic_multiplier - value / slope;
            plastic_multiplier = if slope < 0.0 && newton > lower && newton < upper {
                newton
            } else {
                0.5 * (lower + upper)
            };
        }
        Ok((
            plastic_deformation_gradient(plastic_multiplier)?,
            equivalent_plastic_strain + Quantity::new(plastic_multiplier),
        )
            .into())
    }
    /// Calculates and returns the algorithmic (consistent) tangent stiffness
    /// associated with the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \frac{\mathrm{d}\mathbf{P}}{\mathrm{d}\mathbf{F}} = \frac{\partial\mathbf{P}}{\partial\mathbf{F}}
    ///   + \frac{\partial\mathbf{P}}{\partial\mathbf{F}_\mathrm{p}}:\frac{\mathrm{d}\mathbf{F}_\mathrm{p}^{n+1}}{\mathrm{d}\mathbf{F}}
    /// ```
    /// Formed by central finite differencing of the return-mapped stress, so it
    /// captures the dependence of the updated plastic state on the total
    /// deformation gradient and restores quadratic convergence of the outer solve.
    fn algorithmic_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        state_variables: &PlasticStateVariables,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let mut tangent = FirstPiolaKirchhoffTangentStiffness::zero();
        for k in 0..3 {
            for l in 0..3 {
                let mut plus = deformation_gradient.clone();
                plus[k][l] += perturbation(0.5 * EPSILON);
                let mut minus = deformation_gradient.clone();
                minus[k][l] -= perturbation(0.5 * EPSILON);
                let stress_plus = self.first_piola_kirchhoff_stress(
                    &plus,
                    &self.return_map(&plus, state_variables)?.0,
                )?;
                let stress_minus = self.first_piola_kirchhoff_stress(
                    &minus,
                    &self.return_map(&minus, state_variables)?.0,
                )?;
                for i in 0..3 {
                    for j in 0..3 {
                        tangent[i][j][k][l] = (stress_plus[i][j] - stress_minus[i][j]) / EPSILON;
                    }
                }
            }
        }
        Ok(tangent)
    }
}

/// Zeroth-order root-finding methods for elastic-plastic solid constitutive models.
pub trait ZerothOrderRoot {
    /// Solve for the unknown components of the deformation gradients under an applied load.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F},\mathbf{F}_\mathrm{p}) - \boldsymbol{\lambda} - \mathbf{P}_0 = \mathbf{0}
    /// ```
    /// The plastic state is updated by a nested return mapping at each load step.
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl ZerothOrderRootFinding<FirstPiolaKirchhoffStress, DeformationGradient>,
    ) -> Result<(Times, DeformationGradients, PlasticStateVariablesHistory), ConstitutiveError>;
}

impl<C> ZerothOrderRoot for C
where
    C: ElasticPlastic,
{
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl ZerothOrderRootFinding<FirstPiolaKirchhoffStress, DeformationGradient>,
    ) -> Result<(Times, DeformationGradients, PlasticStateVariablesHistory), ConstitutiveError>
    {
        let (matrix, prescribed, time) = bcs(applied_load);
        let mut vector = Vector::zero(matrix.len());
        let mut state = self.initial_state();
        let mut deformation_gradient = DeformationGradient::identity();
        let mut deformation_gradients = vec![deformation_gradient.clone()];
        let mut states = vec![state.clone()];
        for time_step in time.iter().skip(1) {
            prescribed
                .iter()
                .for_each(|(index, function)| vector[*index] = function(*time_step));
            let previous_state = state.clone();
            deformation_gradient = solver
                .root(
                    |deformation_gradient: &DeformationGradient| {
                        let updated_state =
                            self.return_map(deformation_gradient, &previous_state)?;
                        Ok(self
                            .first_piola_kirchhoff_stress(deformation_gradient, &updated_state.0)?)
                    },
                    deformation_gradient.clone(),
                    EqualityConstraint::Linear(matrix.clone(), vector.clone()),
                )
                .map_err(|error| ConstitutiveError::upstream(error, self))?;
            state = self.return_map(&deformation_gradient, &previous_state)?;
            deformation_gradients.push(deformation_gradient.clone());
            states.push(state.clone());
        }
        Ok((
            time.iter().copied().collect(),
            deformation_gradients.into(),
            states.into(),
        ))
    }
}

/// First-order root-finding methods for elastic-plastic solid constitutive models.
pub trait FirstOrderRoot {
    /// Solve for the unknown components of the deformation gradients under an applied load.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F},\mathbf{F}_\mathrm{p}) - \boldsymbol{\lambda} - \mathbf{P}_0 = \mathbf{0}
    /// ```
    /// The plastic state is updated by a nested return mapping at each load step, and
    /// the algorithmic (consistent) tangent is supplied to the solver.
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
    ) -> Result<(Times, DeformationGradients, PlasticStateVariablesHistory), ConstitutiveError>;
}

impl<C> FirstOrderRoot for C
where
    C: ElasticPlastic,
{
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
    ) -> Result<(Times, DeformationGradients, PlasticStateVariablesHistory), ConstitutiveError>
    {
        let (matrix, prescribed, time) = bcs(applied_load);
        let mut vector = Vector::zero(matrix.len());
        let mut state = self.initial_state();
        let mut deformation_gradient = DeformationGradient::identity();
        let mut deformation_gradients = vec![deformation_gradient.clone()];
        let mut states = vec![state.clone()];
        for time_step in time.iter().skip(1) {
            prescribed
                .iter()
                .for_each(|(index, function)| vector[*index] = function(*time_step));
            let previous_state = state.clone();
            deformation_gradient = solver
                .root(
                    |deformation_gradient: &DeformationGradient| {
                        let updated_state =
                            self.return_map(deformation_gradient, &previous_state)?;
                        Ok(self
                            .first_piola_kirchhoff_stress(deformation_gradient, &updated_state.0)?)
                    },
                    |deformation_gradient: &DeformationGradient| {
                        Ok(self
                            .algorithmic_tangent_stiffness(deformation_gradient, &previous_state)?)
                    },
                    deformation_gradient.clone(),
                    EqualityConstraint::Linear(matrix.clone(), vector.clone()),
                    None,
                )
                .map_err(|error| ConstitutiveError::upstream(error, self))?;
            state = self.return_map(&deformation_gradient, &previous_state)?;
            deformation_gradients.push(deformation_gradient.clone());
            states.push(state.clone());
        }
        Ok((
            time.iter().copied().collect(),
            deformation_gradients.into(),
            states.into(),
        ))
    }
}

/// Local block variable / residual for the monolithic solve: the plastic multiplier
/// increment $`\Delta\gamma`$ lives in slot `[0][0]` of a rank-2 container so it fits
/// the rank-2 block-solver interface; the other eight components are pinned to zero.
type PlasticMultiplierBlock = DeformationGradientPlastic;

/// Tangent blocks $`(K_{uu}, K_{vu}, K_{uv}, K_{vv})`$ in the order the block solver takes them.
type MonolithicTangents = (
    FirstPiolaKirchhoffTangentStiffness,
    TensorRank4<3, Intermediate, Reference, Current, Reference, Dimensionless>,
    TensorRank4<3, Current, Reference, Intermediate, Reference, Stress>,
    TensorRank4<3, Intermediate, Reference, Intermediate, Reference, Dimensionless>,
);

/// The Fischer-Burmeister complementarity function.
///
/// ```math
/// \varphi(a, b) = a + b - \sqrt{a^2 + b^2}, \qquad \varphi(a,b) = 0 \iff a \geq 0,\ b \geq 0,\ ab = 0
/// ```
fn fischer_burmeister(a: Scalar, b: Scalar) -> Scalar {
    a + b - (a * a + b * b).sqrt()
}

/// Monolithic (block) root-finding methods for elastic-plastic solid constitutive models.
pub trait MonolithicRoot {
    /// Solve for the unknown components of the deformation gradients under an applied load,
    /// stepping the deformation gradient and the plastic multiplier increment together.
    ///
    /// The yield inequality is imposed by a Fischer-Burmeister complementarity residual, so
    /// every load step goes through the same block solve and elastic steps recover
    /// $`\Delta\gamma = 0`$ on their own. The flow direction is frozen at the start-of-step
    /// trial state. `strategy` selects the block linear solve
    /// ([`SolveStrategy::Condensed`] or [`SolveStrategy::Monolithic`]).
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderRootFindingBlock<
            DeformationGradient,
            PlasticMultiplierBlock,
            FirstPiolaKirchhoffStress,
            PlasticMultiplierBlock,
            FirstPiolaKirchhoffTangentStiffness,
            TensorRank4<3, Intermediate, Reference, Current, Reference, Dimensionless>,
            TensorRank4<3, Current, Reference, Intermediate, Reference, Stress>,
            TensorRank4<3, Intermediate, Reference, Intermediate, Reference, Dimensionless>,
        >,
        strategy: SolveStrategy,
    ) -> Result<(Times, DeformationGradients, PlasticStateVariablesHistory), ConstitutiveError>;
}

impl<C> MonolithicRoot for C
where
    C: ElasticPlastic,
{
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderRootFindingBlock<
            DeformationGradient,
            PlasticMultiplierBlock,
            FirstPiolaKirchhoffStress,
            PlasticMultiplierBlock,
            FirstPiolaKirchhoffTangentStiffness,
            TensorRank4<3, Intermediate, Reference, Current, Reference, Dimensionless>,
            TensorRank4<3, Current, Reference, Intermediate, Reference, Stress>,
            TensorRank4<3, Intermediate, Reference, Intermediate, Reference, Dimensionless>,
        >,
        strategy: SolveStrategy,
    ) -> Result<(Times, DeformationGradients, PlasticStateVariablesHistory), ConstitutiveError>
    {
        let (matrix, prescribed, time) = bcs(applied_load);
        let mut global_pattern = Vec::new();
        for row in 0..matrix.len() {
            for column in 0..9 {
                if matrix[row][column] != 0.0 {
                    global_pattern.push((row, column));
                }
            }
        }
        let mut global_matrix = CscMatrix::from_pattern(matrix.len(), 9, global_pattern);
        global_matrix.fill(|_, _| 1.0);
        let mut global_vector = Vector::zero(matrix.len());
        let local_pattern: Vec<(usize, usize)> = (0..8).map(|row| (row, row + 1)).collect();
        let mut local_matrix = CscMatrix::from_pattern(8, 9, local_pattern);
        local_matrix.fill(|_, _| 1.0);
        let local_constraint = (local_matrix, Vector::zero(8));

        let reference_yield_stress = self.initial_yield_stress().value();
        let mut state = self.initial_state();
        let mut deformation_gradient = DeformationGradient::identity();
        let mut deformation_gradients = vec![deformation_gradient.clone()];
        let mut states = vec![state.clone()];
        for time_step in time.iter().skip(1) {
            prescribed
                .iter()
                .for_each(|(index, function)| global_vector[*index] = function(*time_step));
            let plastic_deformation_gradient_previous = state.0.clone();
            let equivalent_plastic_strain_previous = state.1;
            let flow_direction = {
                let deviatoric = self
                    .mandel_stress(
                        &deformation_gradient,
                        &plastic_deformation_gradient_previous,
                    )?
                    .deviatoric();
                let direction = self.flow_direction(&deviatoric)?;
                (&direction + direction.transpose()) * 0.5
            };
            let plastic_deformation_gradient = |plastic_multiplier: Scalar| {
                (&flow_direction * plastic_multiplier)
                    .expm()
                    .map(|increment| increment * &plastic_deformation_gradient_previous)
                    .map_err(|error| ConstitutiveError::custom(format!("{error:?}"), self))
            };
            let scaled_yield_function = |global: &DeformationGradient,
                                         plastic_multiplier: Scalar|
             -> Result<Scalar, ConstitutiveError> {
                let deviatoric = self
                    .mandel_stress(global, &plastic_deformation_gradient(plastic_multiplier)?)?
                    .deviatoric();
                Ok(self
                    .yield_function(
                        &deviatoric,
                        equivalent_plastic_strain_previous + Quantity::new(plastic_multiplier),
                    )?
                    .value()
                    / reference_yield_stress)
            };
            let residual_global =
                |global: &DeformationGradient,
                 local: &PlasticMultiplierBlock|
                 -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
                    let plastic = plastic_deformation_gradient(local[0][0].value())?;
                    self.first_piola_kirchhoff_stress(global, &plastic)
                };
            let residual_local = |global: &DeformationGradient, local: &PlasticMultiplierBlock| {
                let plastic_multiplier = local[0][0].value();
                let scaled = scaled_yield_function(global, plastic_multiplier)?;
                let mut residual = PlasticMultiplierBlock::zero();
                residual[0][0] = Quantity::new(fischer_burmeister(plastic_multiplier, -scaled));
                for i in 0..3 {
                    for j in 0..3 {
                        if i != 0 || j != 0 {
                            residual[i][j] = local[i][j];
                        }
                    }
                }
                Ok::<_, ConstitutiveError>(residual)
            };
            let tangents = |global: &DeformationGradient,
                            local: &PlasticMultiplierBlock|
             -> Result<MonolithicTangents, ConstitutiveError> {
                let plastic_multiplier = local[0][0].value();
                let k_uu = self.first_piola_kirchhoff_tangent_stiffness(
                    global,
                    &plastic_deformation_gradient(plastic_multiplier)?,
                )?;
                let stress_slope = (self.first_piola_kirchhoff_stress(
                    global,
                    &plastic_deformation_gradient(plastic_multiplier + 0.5 * EPSILON)?,
                )? - self.first_piola_kirchhoff_stress(
                    global,
                    &plastic_deformation_gradient(plastic_multiplier - 0.5 * EPSILON)?,
                )?) / EPSILON;
                let mut k_uv =
                    TensorRank4::<3, Current, Reference, Intermediate, Reference, Stress>::zero();
                for i in 0..3 {
                    for j in 0..3 {
                        k_uv[i][j][0][0] = stress_slope[i][j];
                    }
                }
                let mut k_vu = TensorRank4::<
                    3,
                    Intermediate,
                    Reference,
                    Current,
                    Reference,
                    Dimensionless,
                >::zero();
                for k in 0..3 {
                    for l in 0..3 {
                        let mut plus = global.clone();
                        plus[k][l] += perturbation(0.5 * EPSILON);
                        let mut minus = global.clone();
                        minus[k][l] -= perturbation(0.5 * EPSILON);
                        let slope = (fischer_burmeister(
                            plastic_multiplier,
                            -scaled_yield_function(&plus, plastic_multiplier)?,
                        ) - fischer_burmeister(
                            plastic_multiplier,
                            -scaled_yield_function(&minus, plastic_multiplier)?,
                        )) / EPSILON;
                        k_vu[0][0][k][l] = Quantity::new(slope);
                    }
                }
                let mut k_vv = TensorRank4::<
                    3,
                    Intermediate,
                    Reference,
                    Intermediate,
                    Reference,
                    Dimensionless,
                >::zero();
                let slope = (fischer_burmeister(
                    plastic_multiplier + 0.5 * EPSILON,
                    -scaled_yield_function(global, plastic_multiplier + 0.5 * EPSILON)?,
                ) - fischer_burmeister(
                    plastic_multiplier - 0.5 * EPSILON,
                    -scaled_yield_function(global, plastic_multiplier - 0.5 * EPSILON)?,
                )) / EPSILON;
                k_vv[0][0][0][0] = Quantity::new(slope);
                for i in 0..3 {
                    for j in 0..3 {
                        if i != 0 || j != 0 {
                            k_vv[i][j][i][j] = Quantity::new(1.0);
                        }
                    }
                }
                Ok((k_uu, k_vu, k_uv, k_vv))
            };
            let (deformation_gradient_new, local_new) = solver
                .root_block(
                    |global, local| {
                        residual_global(global, local).map_err(|e: ConstitutiveError| e.to_string())
                    },
                    |global, local| {
                        residual_local(global, local).map_err(|e: ConstitutiveError| e.to_string())
                    },
                    |global, local| {
                        tangents(global, local).map_err(|e: ConstitutiveError| e.to_string())
                    },
                    (deformation_gradient.clone(), PlasticMultiplierBlock::zero()),
                    (global_matrix.clone(), global_vector.clone()),
                    local_constraint.clone(),
                    None,
                    strategy.clone(),
                )
                .map_err(|error| ConstitutiveError::upstream(error, self))?;
            let plastic_multiplier = local_new[0][0].value();
            let plastic_deformation_gradient_new =
                plastic_deformation_gradient(plastic_multiplier)?;
            deformation_gradient = deformation_gradient_new;
            state = (
                plastic_deformation_gradient_new,
                equivalent_plastic_strain_previous + Quantity::new(plastic_multiplier),
            )
                .into();
            deformation_gradients.push(deformation_gradient.clone());
            states.push(state.clone());
        }
        Ok((
            time.iter().copied().collect(),
            deformation_gradients.into(),
            states.into(),
        ))
    }
}
