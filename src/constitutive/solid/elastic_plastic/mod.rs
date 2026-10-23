//! Elastic-plastic solid constitutive models.

mod canonical;
pub(crate) mod coupled;
mod load;
mod root;
mod step_error;

pub use root::Root;

use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::plastic::{PlasticHardening, PlasticStateVariables, RateIndependentPlastic},
        solid::Solid,
    },
    math::{
        ContractFirstSecondWithSecond, ContractSecondWithFirst, IDENTITY, Matrix, Quantity, Rank2,
        integrate::StepError, optimize::NewtonRaphson,
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, DeformationGradient, DeformationGradientPlastic,
        FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness, MandelStressElastic,
        Scalar, SecondPiolaKirchhoffStress, SecondPiolaKirchhoffTangentStiffness,
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
    Self: Solid + PlasticHardening,
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
    /// surface, a plastic corrector. The corrector is the fully implicit step,
    /// ```math
    /// \mathbf{F}_\mathrm{p}^{n+1} = \exp(\Delta\gamma\,\mathbf{N}^{n+1})\cdot\mathbf{F}_\mathrm{p}^{n},
    /// ```
    /// with the flow direction $`\mathbf{N}^{n+1}`$ taken at the end of the step, solved
    /// together with $`\Delta\gamma`$ by a coupled Newton iteration on
    /// $`(\Delta\gamma\,\mathbf{N},\Delta\gamma)`$. The exponential map is unimodular
    /// for the trace-free $`\mathbf{N}`$ and so needs no step limit, and, unlike a
    /// direction frozen at the trial state, the solve is exact for any elastic model
    /// and does not depend on the step size to be solvable.
    fn return_map(
        &self,
        deformation_gradient: &DeformationGradient,
        state_variables: &PlasticStateVariables,
    ) -> Result<PlasticStateVariables, ConstitutiveError> {
        let converged = coupled::solve(
            self,
            deformation_gradient,
            state_variables,
            &NewtonRaphson::default(),
        )?;
        Ok(coupled::updated_state(state_variables, converged.as_ref()))
    }
    /// The equivalent strain at which the response yields, the extent of its reversible
    /// range, at the equivalent plastic strain of the state.
    ///
    /// ```math
    /// \varepsilon_\mathrm{y} = \frac{\sigma_\mathrm{y}(\bar\varepsilon_\mathrm{p})}{3\mu}
    /// ```
    fn yield_strain(&self, state: &PlasticStateVariables) -> Result<Quantity, ConstitutiveError> {
        let (_, &equivalent_plastic_strain) = state.into();
        Ok(self
            .yield_stress(equivalent_plastic_strain)?
            .ratio(self.shear_modulus() * 3.0))
    }
    /// The error indicators of one load step, from the deformation gradients and the plastic
    /// states at its two ends.
    ///
    /// The return map is exact for any step when the flow direction is constant, so the
    /// step is resolved by how much the direction changes between its two ends, and by
    /// whether it runs past the onset of yielding.
    ///
    /// The error of the step in the plastic deformation is estimated by the change of the
    /// flow direction over it, weighted by the plastic increment.
    ///
    /// ```math
    /// \tfrac{1}{2}\,\Delta\gamma\,\|\mathbf{N}^{n+1} - \mathbf{N}^{n}\|
    /// ```
    ///
    /// The step takes the increment at the direction at its end, and the exact one is about
    /// that at the mean of the two directions, so the error of the step is half of their
    /// difference. It is that error as the step shrinks, and more than it for larger steps.
    ///
    /// The directions are those at the two ends of the step, so it measures the turning
    /// of the load path as well as of the flow. A direction frozen at the trial state
    /// instead would miss the turning of the load, and vanishes identically for a
    /// radial return such as Hencky with a von Mises surface.
    ///
    /// It vanishes when the direction is constant, as under proportional loading, however
    /// large the step, and falls as the square of the step otherwise.
    ///
    /// The overshoot is the plastic increment $`\Delta\gamma`$ of a step that begins inside
    /// the yield surface, how far the step runs past the onset of yielding, and zero
    /// otherwise.
    fn step_error(
        &self,
        deformation_gradient_old: &DeformationGradient,
        deformation_gradient: &DeformationGradient,
        state_old: &PlasticStateVariables,
        state_new: &PlasticStateVariables,
    ) -> Result<StepError, ConstitutiveError> {
        step_error::evaluate(
            self,
            deformation_gradient_old,
            deformation_gradient,
            state_old,
            state_new,
        )
    }
    /// The first Piola-Kirchhoff stress, the consistent tangent stiffness and the updated
    /// plastic state of one load step, from one local solve.
    ///
    /// The local unknowns $`(\mathbf{E},\Delta\gamma)`$ of the step are converged by the
    /// local solver of [`SolveStrategy::Condensed`](crate::math::optimize::SolveStrategy::Condensed) and eliminated from the tangent by a
    /// Schur complement, so a caller needing both the force and the stiffness pays for
    /// one solve.
    fn condensed(
        &self,
        deformation_gradient: &DeformationGradient,
        state_variables: &PlasticStateVariables,
        local_solver: &NewtonRaphson,
    ) -> Result<
        (
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            PlasticStateVariables,
        ),
        ConstitutiveError,
    > {
        coupled::condensed(self, deformation_gradient, state_variables, local_solver)
    }
}

/// The Fischer-Burmeister complementarity function.
///
/// ```math
/// \varphi(a, b) = a + b - \sqrt{a^2 + b^2}, \qquad \varphi(a,b) = 0 \iff a \geq 0,\ b \geq 0,\ ab = 0
/// ```
pub(crate) fn fischer_burmeister(a: Scalar, b: Scalar) -> Scalar {
    a + b - (a * a + b * b).sqrt()
}
