use crate::{
    constitutive::{ConstitutiveError, solid::Solid},
    math::{
        ContractFirstSecondIndicesWithSecondIndicesOf, ContractSecondIndexWithFirstIndexOf,
        IDENTITY, Matrix, Rank2, Tensor, TensorArray, TensorVec, Vector,
        integrate::ExplicitIV,
        optimize::{
            EqualityConstraint, FirstOrderRootFinding, OptimizationError, ZerothOrderRootFinding,
        },
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient,
        DeformationGradientPlastic, DeformationGradientRatePlastic,
        DeformationGradientRatesPlastic, DeformationGradients, DeformationGradientsPlastic,
        FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness, MandelStressElastic,
        Scalar, SecondPiolaKirchhoffStress, SecondPiolaKirchhoffTangentStiffness,
        StretchingRatePlastic, Times,
    },
};

/// Required methods for plastic constitutive models.
pub trait Plastic {
    /// Returns the initial yield stress.
    fn initial_yield_stress(&self) -> Scalar;
    /// Returns the isotropic hardening slope.
    fn hardening_slope(&self) -> Scalar;
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
            if self.hardening_slope() != 0.0 {
                todo!("Need to integrate dY/dt = H * eqps, where eqps = sqrt(2/3) * |D_p|");
            }
            Ok(deviatoric_mandel_stress_e
                * (self.reference_flow_rate() / magnitude
                    * (magnitude / self.initial_yield_stress())
                        .powf(1.0 / self.rate_sensitivity())))
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

/// First-order root-finding methods for elastic-plastic constitutive models.
pub trait FirstOrderRoot {
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
        solver: impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
    ) -> Result<(Times, DeformationGradients, DeformationGradientsPlastic), ConstitutiveError>;
    #[doc(hidden)]
    fn root_inner_1(
        &self,
        deformation_gradient_p: &DeformationGradientPlastic,
        equality_constraint: EqualityConstraint,
        solver: &impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
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
        match match applied_load {
            AppliedLoad::UniaxialStress(deformation_gradient_11, time) => {
                let mut matrix = Matrix::zero(4, 9);
                let mut vector = Vector::zero(4);
                matrix[0][0] = 1.0;
                matrix[1][1] = 1.0;
                matrix[2][2] = 1.0;
                matrix[3][5] = 1.0;
                integrator.integrate(
                    |_: Scalar,
                     deformation_gradient_p: &DeformationGradientPlastic,
                     deformation_gradient: &DeformationGradient| {
                        Ok(self.plastic_deformation_gradient_rate(
                            deformation_gradient,
                            deformation_gradient_p,
                        )?)
                    },
                    |t: Scalar,
                     deformation_gradient_p: &DeformationGradientPlastic,
                     deformation_gradient: &DeformationGradient| {
                        vector[0] = deformation_gradient_11(t);
                        Ok(self.root_inner_0(
                            deformation_gradient_p,
                            EqualityConstraint::Linear(matrix.clone(), vector.clone()),
                            &solver,
                            deformation_gradient,
                        )?)
                    },
                    time,
                    DeformationGradientPlastic::identity(),
                    DeformationGradient::identity(),
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

impl<T> FirstOrderRoot for T
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
        solver: impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
    ) -> Result<(Times, DeformationGradients, DeformationGradientsPlastic), ConstitutiveError> {
        match match applied_load {
            AppliedLoad::UniaxialStress(deformation_gradient_11, time) => {
                let mut matrix = Matrix::zero(4, 9);
                let mut vector = Vector::zero(4);
                matrix[0][0] = 1.0;
                matrix[1][1] = 1.0;
                matrix[2][2] = 1.0;
                matrix[3][5] = 1.0;
                integrator.integrate(
                    |_: Scalar,
                     deformation_gradient_p: &DeformationGradientPlastic,
                     deformation_gradient: &DeformationGradient| {
                        Ok(self.plastic_deformation_gradient_rate(
                            deformation_gradient,
                            deformation_gradient_p,
                        )?)
                    },
                    |t: Scalar,
                     deformation_gradient_p: &DeformationGradientPlastic,
                     deformation_gradient: &DeformationGradient| {
                        vector[0] = deformation_gradient_11(t);
                        Ok(self.root_inner_1(
                            deformation_gradient_p,
                            EqualityConstraint::Linear(matrix.clone(), vector.clone()),
                            &solver,
                            deformation_gradient,
                        )?)
                    },
                    time,
                    DeformationGradientPlastic::identity(),
                    DeformationGradient::identity(),
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
    fn root_inner_1(
        &self,
        deformation_gradient_p: &DeformationGradientPlastic,
        equality_constraint: EqualityConstraint,
        solver: &impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
        initial_guess: &DeformationGradient,
    ) -> Result<DeformationGradient, OptimizationError> {
        solver.root(
            |deformation_gradient: &DeformationGradient| {
                Ok(self
                    .first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?)
            },
            |deformation_gradient: &DeformationGradient| {
                Ok(self.first_piola_kirchhoff_tangent_stiffness(
                    deformation_gradient,
                    deformation_gradient_p,
                )?)
            },
            initial_guess.clone(),
            equality_constraint,
        )
    }
}
