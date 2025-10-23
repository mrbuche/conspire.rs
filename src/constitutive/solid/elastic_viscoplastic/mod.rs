//! Elastic-viscoplastic constitutive models.

use crate::{
    constitutive::{ConstitutiveError, solid::Solid},
    math::{
        ContractFirstSecondIndicesWithSecondIndicesOf, ContractSecondIndexWithFirstIndexOf,
        IDENTITY, Matrix, Rank2, Tensor, TensorArray, TensorTuple, TensorTupleVec, Vector,
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

/// ???
pub type StateVariables = TensorTuple<DeformationGradientPlastic, Scalar>;

/// ???
pub type StateVariablesHistory = TensorTupleVec<DeformationGradientPlastic, Scalar>;

/// Required methods for plastic constitutive models.
pub trait Plastic {
    /// Returns the initial yield stress.
    fn initial_yield_stress(&self) -> &Scalar;
    /// Returns the isotropic hardening slope.
    fn hardening_slope(&self) -> &Scalar;
}

/// Required methods for viscoplastic constitutive models.
pub trait Viscoplastic
where
    Self: Plastic,
{
    /// Returns the rate_sensitivity parameter.
    fn rate_sensitivity(&self) -> &Scalar;
    /// Returns the reference flow rate.
    fn reference_flow_rate(&self) -> &Scalar;
}

/// Possible applied loads.
pub enum AppliedLoad<'a> {
    /// Uniaxial stress given $`F_{11}`$.
    UniaxialStress(fn(Scalar) -> Scalar, &'a [Scalar]),
    // /// Biaxial stress given $`F_{11}`$ and $`F_{22}`$.
    // BiaxialStress(fn(Scalar) -> Scalar, fn(Scalar) -> Scalar, &'a [Scalar]),
}

/// Required methods for elastic-viscoplastic constitutive models.
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
        todo!("plastic_deformation_gradient_rate() deprecated");
        let jacobian = deformation_gradient.jacobian().unwrap();
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let cauchy_stress = self.cauchy_stress(deformation_gradient, deformation_gradient_p)?;
        let mandel_stress_e = (deformation_gradient_e.transpose()
            * cauchy_stress
            * deformation_gradient_e.inverse_transpose())
            * jacobian;
        Ok(self
            .plastic_stretching_rate(mandel_stress_e.deviatoric(), self.initial_yield_stress())?
            * deformation_gradient_p)
    }
    /// Calculates and returns the rate of plastic stretching.
    ///
    /// ```math
    /// \mathbf{D}_\mathrm{p} = d_0\left(\frac{|\mathbf{M}_\mathrm{e}'|}{Y(S)}\right)^{\footnotesize\tfrac{1}{m}}\frac{\mathbf{M}_\mathrm{e}'}{|\mathbf{M}_\mathrm{e}'|}
    /// ```
    fn plastic_stretching_rate(
        &self,
        deviatoric_mandel_stress_e: MandelStressElastic,
        yield_stress: &Scalar,
    ) -> Result<StretchingRatePlastic, ConstitutiveError> {
        let magnitude = deviatoric_mandel_stress_e.norm();
        if magnitude == 0.0 {
            Ok(StretchingRatePlastic::zero())
        } else {
            Ok(deviatoric_mandel_stress_e
                * (self.reference_flow_rate() / magnitude
                    * (magnitude / yield_stress).powf(1.0 / self.rate_sensitivity())))
        }
    }
    /// Calculates and returns the evolution of the yield stress.
    ///
    /// ```math
    /// \dot{Y} = \sqrt{\frac{2}{3}}\,H\,|\mathbf{D}_\mathrm{p}|
    /// ```
    fn yield_stress_evolution(
        &self,
        plastic_stretching_rate: &StretchingRatePlastic,
    ) -> Result<Scalar, ConstitutiveError> {
        Ok(self.hardening_slope() * plastic_stretching_rate.norm() * (2.0_f64 / 3.0).sqrt())
    }
    /// ???
    ///
    /// ```math
    /// \dot{\mathbf{F}}_\mathrm{p} = \mathbf{D}_\mathrm{p}\cdot\mathbf{F}_\mathrm{p}
    /// ```
    fn foo(
        &self,
        deformation_gradient: &DeformationGradient,
        state_variables: &StateVariables,
    ) -> Result<StateVariables, ConstitutiveError> {
        //
        // Some redundancy with above `plastic_deformation_gradient_rate`` function.
        // Since the above equation is always true, maybe delete that function and use this instead.
        // The yield stress evolution on the other hand could be different, so keep that function.
        //
        let (deformation_gradient_p, yield_stress) = state_variables.into();
        let jacobian = deformation_gradient.jacobian().unwrap();
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let cauchy_stress = self.cauchy_stress(deformation_gradient, deformation_gradient_p)?;
        let mandel_stress_e = (deformation_gradient_e.transpose()
            * cauchy_stress
            * deformation_gradient_e.inverse_transpose())
            * jacobian;
        let plastic_stretching_rate =
            self.plastic_stretching_rate(mandel_stress_e.deviatoric(), yield_stress)?;
        //
        // You have tensor rank N list/vec and 2D and so on.
        // Could you combine these into a single impl?
        // Like TensorList<T> or TensorVec<T> where T is the tensor type.
        // Might be able to even automatically handle the nested cases with T=TensorList<...>.
        //
        // And also see if you can do something with (T1, T2) being a tensor?
        // The thing above would help you automatically handle a Vec of these tuple.
        // Integration impls could handle tuples automatically if they have the tensor impls.
        //
        // Also note that you are going to have to this and `plastic_stretching_rate` a function of `yield_stress`.
        //
        // Is this going to get out of hand later? With models with ISVs that are arbitrary?
        // Do you want to start now handling that in a consistent way?
        // Which will help you implement only 1 solver/etc. in constitutive/ and fem/ for ISV models?
        // (Maybe still separate for explicitly- and implicitly-rate-dependent models).
        // Could make a trait in solid/ for all implicitly-rate-dependent models
        // which has an evolution function accepting the state and returning the evolution.
        // The i/o here could be a sized array (method would have to flatten and stuff)
        // or generic parameters that impl certain methods (could be nice),
        // since the fem/ would be generic over them just like integrate/ would be.
        // The mod could be internal_state_variables and have the two submodules.
        //
        // Maybe not though, that is somewhat further in the future,
        // and mainly it might be nice to think about how to collect things in fem/
        // like the plastic deformation as [[Fp; integ_pts]; elements] and so on,
        // and hard to know what these are if just a general state.
        // Maybe just do it the specific way for now? Can always refactor later.
        // And use the nice parts of the specific impl to drive how to get general one working just as nice later.
        //
        // Fp_dot = f_1(F, Fp, Y) and Y_dot = f_2(F, Fp, Y)
        // (Fp_dot, Y_dot) = (f_1(F, Fp, Y), f_2(F, Fp, Y))
        // (dXdt, dYdt, ...) = (f_1(Z0; X, Y, ...), f_2(Z0; X, Y, ...), ...)
        // should handle nicely in integrate/ if Tensor/etc. is implemented for the tuples
        // fem/ has Vec<[(F, Fp, Y); N]> which should still be alright
        // also note that each integration point evolution equation only depends on the local (F, Fp, Y)
        // is there a way to get integrate/ to handle that simplification for performance purposes?
        // Would compute all the slopes/steps separately, but the error together.
        // Perhaps not actually worth it, since the IV solve for z needs all variables anyway.
        //
        Ok(StateVariables::from((
            &plastic_stretching_rate * deformation_gradient_p,
            self.yield_stress_evolution(&plastic_stretching_rate)?,
        )))
    }
}

/// Zeroth-order root-finding methods for elastic-viscoplastic constitutive models.
pub trait ZerothOrderRoot {
    /// Solve for the unknown components of the deformation gradients under an applied load.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F},\mathbf{F}_\mathrm{p}) - \boldsymbol{\lambda} - \mathbf{P}_0 = \mathbf{0}
    /// ```
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

/// First-order root-finding methods for elastic-viscoplastic constitutive models.
pub trait FirstOrderRoot {
    /// Solve for the unknown components of the deformation gradients under an applied load.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F},\mathbf{F}_\mathrm{p}) - \boldsymbol{\lambda} - \mathbf{P}_0 = \mathbf{0}
    /// ```
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ExplicitIV<
            StateVariables,
            DeformationGradient,
            StateVariablesHistory,
            DeformationGradients,
        >,
        solver: impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
    ) -> Result<(Times, DeformationGradients, StateVariablesHistory), ConstitutiveError>;
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
            StateVariables,
            DeformationGradient,
            StateVariablesHistory,
            DeformationGradients,
        >,
        solver: impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
    ) -> Result<(Times, DeformationGradients, StateVariablesHistory), ConstitutiveError> {
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
                     state_variables: &StateVariables,
                     deformation_gradient: &DeformationGradient| {
                        Ok(self.foo(deformation_gradient, state_variables)?)
                    },
                    |t: Scalar,
                     state_variables: &StateVariables,
                     deformation_gradient: &DeformationGradient| {
                        let (deformation_gradient_p, _) = state_variables.into();
                        vector[0] = deformation_gradient_11(t);
                        Ok(self.root_inner_1(
                            deformation_gradient_p,
                            EqualityConstraint::Linear(matrix.clone(), vector.clone()),
                            &solver,
                            deformation_gradient,
                        )?)
                    },
                    time,
                    StateVariables::from((
                        DeformationGradientPlastic::identity(),
                        *self.initial_yield_stress(),
                    )),
                    DeformationGradient::identity(),
                )
            }
        } {
            Ok((times, state_variables, _, deformation_gradients)) => {
                Ok((times, deformation_gradients, state_variables))
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
