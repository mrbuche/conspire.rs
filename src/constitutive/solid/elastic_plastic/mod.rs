//! Elastic-plastic solid constitutive models.

mod canonical;

use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::plastic::{
            Plastic, PlasticStateVariables, PlasticStateVariablesHistory, RateIndependentPlastic,
        },
        solid::Solid,
    },
    math::{
        ContractFirstSecondWithSecond, ContractSecondWithFirst, IDENTITY, Matrix, Quantity, Rank2,
        TensorArray, Vector,
        optimize::{EqualityConstraint, FirstOrderRootFinding, ZerothOrderRootFinding},
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, DeformationGradient, DeformationGradientPlastic,
        DeformationGradients, FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness,
        FlowDirectionPlastic, MandelStressElastic, Scalar, SecondPiolaKirchhoffStress,
        SecondPiolaKirchhoffTangentStiffness, Times,
    },
    units::Time,
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
    /// via an elastic predictor and, if the trial state lies outside the yield
    /// surface, a plastic corrector for the incremental multiplier $`\Delta\gamma`$
    /// with the flow direction frozen at the trial state.
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
        let flow_direction = self.flow_direction(&deviatoric_trial)?;
        let plastic_deformation_gradient = |plastic_multiplier: Scalar| {
            (FlowDirectionPlastic::identity() - &flow_direction * plastic_multiplier).inverse()
                * deformation_gradient_p
        };
        let residual = |plastic_multiplier: Scalar| -> Result<Scalar, ConstitutiveError> {
            let deviatoric = self
                .mandel_stress(
                    deformation_gradient,
                    &plastic_deformation_gradient(plastic_multiplier),
                )?
                .deviatoric();
            Ok(self
                .yield_function(
                    &deviatoric,
                    equivalent_plastic_strain + Quantity::new(plastic_multiplier),
                )?
                .value())
        };
        // The flow direction is deviatoric, so `|N| = 1` bounds its eigenvalues
        // below `sqrt(2/3)`; keeping `dg <= 0.9` keeps `I - dg N` positive definite
        // and hence `F_p` invertible even for a wildly off-equilibrium trial state
        // handed in by the outer iteration.
        let (mut low, mut high) = (0.0, 1e-3);
        while high < 0.9 && residual(high)? > 0.0 {
            high = (2.0 * high).min(0.9);
        }
        for _ in 0..64 {
            let midpoint = 0.5 * (low + high);
            if residual(midpoint)? > 0.0 {
                low = midpoint
            } else {
                high = midpoint
            }
        }
        let plastic_multiplier = 0.5 * (low + high);
        Ok((
            plastic_deformation_gradient(plastic_multiplier),
            equivalent_plastic_strain + Quantity::new(plastic_multiplier),
        )
            .into())
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
    /// The plastic state is updated by a nested return mapping at each load step. The
    /// continuum tangent (at fixed plastic state) is supplied to the solver; the
    /// consistent algorithmic tangent is not yet formed, so convergence is not quadratic.
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
                        let updated_state =
                            self.return_map(deformation_gradient, &previous_state)?;
                        Ok(self.first_piola_kirchhoff_tangent_stiffness(
                            deformation_gradient,
                            &updated_state.0,
                        )?)
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
