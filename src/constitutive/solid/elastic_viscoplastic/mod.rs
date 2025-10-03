use crate::{
    constitutive::{ConstitutiveError, solid::Solid},
    math::{
        Matrix, Rank2, Tensor, TensorArray, TensorVec, Vector,
        integrate::ExplicitIV,
        optimize::{EqualityConstraint, OptimizationError, ZerothOrderRootFinding},
    },
    mechanics::{
        CauchyStress, Deformation, DeformationGradient, DeformationGradientPlastic,
        DeformationGradientRatePlastic, DeformationGradientRatesPlastic, DeformationGradients,
        DeformationGradientsPlastic, FirstPiolaKirchhoffStress, MandelStressElastic, Scalar,
        SecondPiolaKirchhoffStress, StretchingRatePlastic, Times,
    },
};

/// Required methods for plastic constitutive models.
pub trait Plastic {
    /// Returns the yield stress.
    fn yield_stress(&self) -> Scalar;
}

/// Required methods for viscoplastic constitutive models.
pub trait Viscoplastic
where
    Self: Plastic,
{
    /// Returns the rate_sensitivity parameter.
    fn rate_sensitivity(&self) -> Scalar;
    /// Returns the reference flow rate.
    fn reference_flow_rate(&self) -> Scalar;
}

/// Possible applied loads.
pub enum AppliedLoad<'a> {
    /// Uniaxial stress given $`F_{11}`$.
    UniaxialStress(fn(Scalar) -> Scalar, &'a [Scalar]),
    // /// Biaxial stress given $`F_{11}`$ and $`F_{22}`$.
    // BiaxialStress(fn(Scalar) -> Scalar, fn(Scalar) -> Scalar, &'a [Scalar]),
}

/// Required methods for elastic-plastic constitutive models.
pub trait ElasticViscoplastic
where
    Self: Solid + Viscoplastic,
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
            Ok(deviatoric_mandel_stress_e
                * (self.reference_flow_rate() / magnitude
                    * (magnitude / self.yield_stress()).powf(1.0 / self.rate_sensitivity())))
        }
    }
}

/// Zeroth-order root-finding methods for elastic-plastic constitutive models.
pub trait ZerothOrderRoot {
    /// ???
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ExplicitIV<
            DeformationGradientRatePlastic,
            DeformationGradient,
            DeformationGradientRatesPlastic,
            DeformationGradients,
        >,
        solver: impl ZerothOrderRootFinding<DeformationGradient>,
    ) -> Result<(Times, DeformationGradients, DeformationGradientsPlastic), ConstitutiveError>;
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
    T: ElasticViscoplastic,
{
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ExplicitIV<
            DeformationGradientRatePlastic,
            DeformationGradient,
            DeformationGradientRatesPlastic,
            DeformationGradients,
        >,
        solver: impl ZerothOrderRootFinding<DeformationGradient>,
    ) -> Result<(Times, DeformationGradients, DeformationGradientsPlastic), ConstitutiveError> {
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
                        Ok((
                            self.plastic_deformation_gradient_rate(
                                &deformation_gradient,
                                deformation_gradient_p,
                            )?,
                            deformation_gradient.clone(),
                        ))
                    },
                    time,
                    DeformationGradientPlastic::identity(),
                )
            }
        } {
            Ok((times, deformation_gradients_p, _, deformation_gradients)) => {
                Ok((times, deformation_gradients, deformation_gradients_p))
            }
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
