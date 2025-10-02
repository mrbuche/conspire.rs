use crate::{
    constitutive::{ConstitutiveError, solid::Solid},
    math::{
        Matrix, Rank2, Tensor, TensorArray, TensorVec, Vector,
        integrate::Explicit,
        optimize::{EqualityConstraint, OptimizationError, ZerothOrderRootFinding},
    },
    mechanics::{
        CauchyStress, Deformation, DeformationGradient, DeformationGradientPlastic,
        DeformationGradientRatePlastic, DeformationGradientRatesPlastic, DeformationGradients,
        DeformationGradientsPlastic, FirstPiolaKirchhoffStress, MandelStressElastic, Scalar,
        SecondPiolaKirchhoffStress, StretchingRatePlastic, Times,
    },
};

/// Possible applied loads.
pub enum AppliedLoad<'a> {
    /// Uniaxial stress given $`F_{11}`$.
    UniaxialStress(fn(Scalar) -> Scalar, &'a [Scalar]),
    // /// Biaxial stress given $`F_{11}`$ and $`F_{22}`$.
    // BiaxialStress(fn(Scalar) -> Scalar, fn(Scalar) -> Scalar, &'a [Scalar]),
}

/// Required methods for elastic-plastic constitutive models.
pub trait ElasticPlastic
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
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyStress, ConstitutiveError> {
        Ok(deformation_gradient
            * self.second_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?
            * deformation_gradient.transpose()
            / deformation_gradient.determinant())
    }
    /// Calculates and returns the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{P} = J\boldsymbol{\sigma}\cdot\mathbf{F}^{-T}
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
    /// Calculates and returns the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{S} = \mathbf{F}^{-1}\cdot\mathbf{P}
    /// ```
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        Ok(deformation_gradient.inverse()
            * self.first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?)
    }
    /// Calculates and returns the rate plastic deformation.
    ///
    /// ```math
    /// \dot{\mathbf{F}}_\mathrm{p} = \mathbf{D}_\mathrm{p}\cdot\mathbf{F}_\mathrm{p}
    /// ```
    fn plastic_deformation_gradient_rate(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<DeformationGradientRatePlastic, ConstitutiveError> {
        let jacobian = deformation_gradient.jacobian().unwrap();
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let cauchy_stress = self.cauchy_stress(deformation_gradient, deformation_gradient_p)?;
        let mandel_stress_e = (deformation_gradient_e.transpose()
            * cauchy_stress
            * deformation_gradient_e.inverse_transpose())
            * jacobian;
        Ok(self.plastic_stretching_rate(mandel_stress_e.deviatoric())? * deformation_gradient_p)
    }
    /// Calculates and returns the rate of plastic stretching.
    ///
    /// ```math
    /// \mathbf{D}_\mathrm{p} = d_0\left(\frac{|\mathbf{M}_\mathrm{e}'|}{Y(S)}\right)^{\footnotesize\tfrac{1}{m}}\frac{\mathbf{M}_\mathrm{e}'}{|\mathbf{M}_\mathrm{e}'|}
    /// ```
    fn plastic_stretching_rate(
        &self,
        deviatoric_mandel_stress_e: MandelStressElastic,
    ) -> Result<StretchingRatePlastic, ConstitutiveError> {
        let magnitude = deviatoric_mandel_stress_e.norm();
        if magnitude == 0.0 {
            Ok(StretchingRatePlastic::zero())
        } else {
            let d_0 = 1e-1; // This should be replaced with a parameter or a function of state
            let m = 0.25; // This should be replaced with a parameter or a function of state
            let y = 3.0; // This should be replaced with a parameter or a function of state
            // also need to do hardening stuff
            Ok(deviatoric_mandel_stress_e * (d_0 / magnitude * (magnitude / y).powf(1.0 / m)))
        }
    }
}

/// Zeroth-order root-finding methods for elastic-plastic constitutive models.
pub trait ZerothOrderRoot {
    /// ???
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl Explicit<DeformationGradientRatePlastic, DeformationGradientRatesPlastic>,
        solver: impl ZerothOrderRootFinding<DeformationGradient>,
    ) -> Result<
        (
            Times,
            DeformationGradients,
            DeformationGradientsPlastic,
            DeformationGradientRatesPlastic,
        ),
        ConstitutiveError,
    >;
    #[doc(hidden)]
    fn root_inner_0(
        &self,
        deformation_gradient_p: &DeformationGradientPlastic,
        equality_constraint: EqualityConstraint,
        solver: &impl ZerothOrderRootFinding<DeformationGradient>,
        initial_guess: &DeformationGradient,
    ) -> Result<DeformationGradient, OptimizationError>;
}

impl<T> ZerothOrderRoot for T
where
    T: ElasticPlastic,
{
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl Explicit<DeformationGradientRatePlastic, DeformationGradientRatesPlastic>,
        solver: impl ZerothOrderRootFinding<DeformationGradient>,
    ) -> Result<
        (
            Times,
            DeformationGradients,
            DeformationGradientsPlastic,
            DeformationGradientRatesPlastic,
        ),
        ConstitutiveError,
    > {
        let mut deformation_gradient = DeformationGradient::identity();
        match match applied_load {
            AppliedLoad::UniaxialStress(deformation_gradient_11, time) => {
                let mut matrix = Matrix::zero(4, 9);
                let mut vector = Vector::zero(4);
                matrix[0][0] = 1.0;
                matrix[1][1] = 1.0;
                matrix[2][2] = 1.0;
                matrix[3][5] = 1.0;
                integrator.integrate(
                    |t: Scalar, deformation_gradient_p: &DeformationGradientPlastic| {
                        vector[0] = deformation_gradient_11(t);
                        deformation_gradient = self.root_inner_0(
                            deformation_gradient_p,
                            EqualityConstraint::Linear(matrix.clone(), vector.clone()),
                            &solver,
                            &deformation_gradient,
                        )?;
                        Ok(self.plastic_deformation_gradient_rate(
                            &deformation_gradient,
                            deformation_gradient_p,
                        )?)
                    },
                    time,
                    DeformationGradientPlastic::identity(),
                )
            }
        } {
            Ok((times, deformation_gradients_p, deformation_gradient_rates_p)) => {
                match applied_load {
                    AppliedLoad::UniaxialStress(deformation_gradient_11, _) => {
                        let mut matrix = Matrix::zero(4, 9);
                        let mut vector = Vector::zero(4);
                        matrix[0][0] = 1.0;
                        matrix[1][1] = 1.0;
                        matrix[2][2] = 1.0;
                        matrix[3][5] = 1.0;
                        match times
                            .iter()
                            .zip(deformation_gradients_p.iter())
                            .map(|(time, deformation_gradient_p)| {
                                vector[0] = deformation_gradient_11(*time);
                                self.root_inner_0(
                                    deformation_gradient_p,
                                    EqualityConstraint::Linear(matrix.clone(), vector.clone()),
                                    &solver,
                                    &deformation_gradient,
                                )
                            })
                            .collect()
                        {
                            Ok(deformation_gradients) => Ok((
                                times,
                                deformation_gradients,
                                deformation_gradients_p,
                                deformation_gradient_rates_p,
                            )),
                            Err(error) => Err(ConstitutiveError::Upstream(
                                format!("{error}"),
                                format!("{self:?}"),
                            )),
                        }
                    }
                }
            }
            // Ok(results) => Ok(results),
            Err(error) => Err(ConstitutiveError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    #[doc(hidden)]
    fn root_inner_0(
        &self,
        deformation_gradient_p: &DeformationGradientPlastic,
        equality_constraint: EqualityConstraint,
        solver: &impl ZerothOrderRootFinding<DeformationGradient>,
        initial_guess: &DeformationGradient,
    ) -> Result<DeformationGradient, OptimizationError> {
        solver.root(
            |deformation_gradient: &DeformationGradient| {
                Ok(self
                    .first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?)
            },
            initial_guess.clone(),
            equality_constraint,
        )
    }
}
