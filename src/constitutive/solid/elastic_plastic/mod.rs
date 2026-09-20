//! Elastic-plastic solid constitutive models.

mod canonical;
mod coupled;

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
        TensorArray, TensorRank2, TensorRank4, Vector,
        optimize::{
            EqualityConstraint, FirstOrderRootFinding, FirstOrderRootFindingBlock, SolveStrategy,
            ZerothOrderRootFinding,
        },
        sparse::CscMatrix,
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, CauchyTangentStiffnessPlastic, DeformationGradient,
        DeformationGradientPlastic, DeformationGradients, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, FirstPiolaKirchhoffTangentStiffnessPlastic,
        MandelStressElastic, MandelStressTangentElastic, MandelStressTangentElasticPlastic, Scalar,
        SecondPiolaKirchhoffStress, SecondPiolaKirchhoffTangentStiffness, Times,
    },
    units::Time,
};
use std::array::from_fn;

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

pub(crate) type Matrix3 = [[Scalar; 3]; 3];
pub(crate) type Entries4 = [[[[Scalar; 3]; 3]; 3]; 3];

pub(crate) fn matrix_3<I, J, U>(tensor: &TensorRank2<3, I, J, U>) -> Matrix3 {
    from_fn(|i| from_fn(|j| tensor[i][j].value()))
}

pub(crate) fn entries_4<I, J, K, L, U>(tensor: &TensorRank4<3, I, J, K, L, U>) -> Entries4 {
    from_fn(|i| from_fn(|j| from_fn(|k| from_fn(|l| tensor[i][j][k][l].value()))))
}

pub(crate) fn rank_4<I, J, K, L, U>(entries: &Entries4) -> TensorRank4<3, I, J, K, L, U> {
    let mut tensor = TensorRank4::zero();
    (0..3).for_each(|i| {
        (0..3).for_each(|j| {
            (0..3).for_each(|k| {
                (0..3).for_each(|l| tensor[i][j][k][l] = Quantity::new(entries[i][j][k][l]))
            })
        })
    });
    tensor
}

//
// M_ij = J F^e_ki sigma_kl F^{e-1}_jl, differentiated once for a direction (a, b) in
// which dF^e_ki = g_ka F^{p-1}_bi: g = 1 for F itself and g = -F^e for F^p, with the
// Jacobian term present only for F.
//
#[allow(clippy::too_many_arguments)]
fn mandel_stress_tangent_entries(
    jacobian: Scalar,
    mandel_stress: &Matrix3,
    cauchy_stress: &Matrix3,
    deformation_gradient_e: &Matrix3,
    deformation_gradient_e_inverse: &Matrix3,
    deformation_gradient_p_inverse: &Matrix3,
    g: &Matrix3,
    d_jacobian: Option<&Matrix3>,
    d_cauchy_stress: &Entries4,
) -> Entries4 {
    let w: Matrix3 = from_fn(|a| {
        from_fn(|j| {
            (0..3)
                .map(|k| {
                    (0..3)
                        .map(|l| {
                            g[k][a] * cauchy_stress[k][l] * deformation_gradient_e_inverse[j][l]
                        })
                        .sum::<Scalar>()
                })
                .sum()
        })
    });
    let v: Matrix3 = from_fn(|j| {
        from_fn(|a| {
            (0..3)
                .map(|m| deformation_gradient_e_inverse[j][m] * g[m][a])
                .sum()
        })
    });
    let u: Matrix3 = from_fn(|i| {
        from_fn(|b| {
            (0..3)
                .map(|n| mandel_stress[i][n] * deformation_gradient_p_inverse[b][n])
                .sum()
        })
    });
    from_fn(|i| {
        from_fn(|j| {
            from_fn(|a| {
                from_fn(|b| {
                    let elastic = (0..3)
                        .map(|k| {
                            (0..3)
                                .map(|l| {
                                    deformation_gradient_e[k][i]
                                        * d_cauchy_stress[k][l][a][b]
                                        * deformation_gradient_e_inverse[j][l]
                                })
                                .sum::<Scalar>()
                        })
                        .sum::<Scalar>();
                    d_jacobian.map_or(0.0, |d_jacobian| d_jacobian[a][b] * mandel_stress[i][j])
                        + jacobian * (deformation_gradient_p_inverse[b][i] * w[a][j] + elastic)
                        - u[i][b] * v[j][a]
                })
            })
        })
    })
}

/// Tangents with respect to the plastic deformation gradient, and of the Mandel
/// stress with respect to both deformation gradients — the pieces a coupled
/// (condensed) return map needs beyond the elastic tangents.
pub trait PlasticTangents
where
    Self: ElasticPlasticOrViscoplastic,
{
    /// Calculates and returns the tangent stiffness of the Cauchy stress with
    /// respect to the plastic deformation gradient.
    ///
    /// ```math
    /// \frac{\partial\sigma_{ij}}{\partial F^\mathrm{p}_{NO}} = -\mathcal{T}^\mathrm{e}_{ijmA} F^\mathrm{e}_{mN} F^{\mathrm{p}-1}_{OA}
    /// ```
    fn cauchy_tangent_stiffness_p(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyTangentStiffnessPlastic, ConstitutiveError>;
    /// Calculates and returns the tangent stiffness of the first Piola-Kirchhoff
    /// stress with respect to the plastic deformation gradient.
    ///
    /// ```math
    /// \frac{\partial P_{iJ}}{\partial F^\mathrm{p}_{NO}} = -\mathcal{C}^\mathrm{e}_{iAmB} F^\mathrm{e}_{mN} F^{\mathrm{p}-1}_{OB} F^{\mathrm{p}-1}_{JA} - P_{iO} F^{\mathrm{p}-1}_{JN}
    /// ```
    fn first_piola_kirchhoff_tangent_stiffness_p(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffTangentStiffnessPlastic, ConstitutiveError>;
    /// Calculates and returns the tangent stiffness of the Mandel stress with
    /// respect to the deformation gradient.
    fn mandel_stress_tangent(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<MandelStressTangentElastic, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        let deformation_gradient_e = deformation_gradient * &deformation_gradient_p_inverse;
        Ok(rank_4(&mandel_stress_tangent_entries(
            self.jacobian(deformation_gradient)?,
            &matrix_3(&self.mandel_stress(deformation_gradient, deformation_gradient_p)?),
            &matrix_3(&self.cauchy_stress(deformation_gradient, deformation_gradient_p)?),
            &matrix_3(&deformation_gradient_e),
            &matrix_3(&deformation_gradient_e.inverse()),
            &matrix_3(&deformation_gradient_p_inverse),
            &matrix_3(&IDENTITY),
            Some(&matrix_3(&deformation_gradient.inverse_transpose())),
            &entries_4(
                &self.cauchy_tangent_stiffness(deformation_gradient, deformation_gradient_p)?,
            ),
        )))
    }
    /// Calculates and returns the tangent stiffness of the Mandel stress with
    /// respect to the plastic deformation gradient.
    fn mandel_stress_tangent_p(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<MandelStressTangentElasticPlastic, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        let deformation_gradient_e = deformation_gradient * &deformation_gradient_p_inverse;
        Ok(rank_4(&mandel_stress_tangent_entries(
            self.jacobian(deformation_gradient)?,
            &matrix_3(&self.mandel_stress(deformation_gradient, deformation_gradient_p)?),
            &matrix_3(&self.cauchy_stress(deformation_gradient, deformation_gradient_p)?),
            &matrix_3(&deformation_gradient_e),
            &matrix_3(&deformation_gradient_e.inverse()),
            &matrix_3(&deformation_gradient_p_inverse),
            &matrix_3(&(&deformation_gradient_e * -1.0)),
            None,
            &entries_4(
                &self.cauchy_tangent_stiffness_p(deformation_gradient, deformation_gradient_p)?,
            ),
        )))
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
        let converged = coupled::solve(self, deformation_gradient, state_variables)?;
        Ok(coupled::updated_state(state_variables, converged.as_ref()))
    }
    /// Return maps one load step and returns the updated plastic state together with the
    /// consistent (algorithmic) first Piola-Kirchhoff tangent stiffness at that state.
    ///
    /// The tangent follows from the implicit function theorem applied to the converged
    /// coupled system, which yields the exact derivative of the stress with respect to
    /// the deformation gradient through the return map; on an elastic step it is just
    /// the continuum tangent. The solve is shared with the state update, so calling this
    /// costs one return mapping.
    fn consistent_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        state_variables: &PlasticStateVariables,
    ) -> Result<(FirstPiolaKirchhoffTangentStiffness, PlasticStateVariables), ConstitutiveError>
    {
        let converged = coupled::solve(self, deformation_gradient, state_variables)?;
        Ok((
            coupled::consistent_tangent(
                self,
                deformation_gradient,
                state_variables,
                converged.as_ref(),
            )?,
            coupled::updated_state(state_variables, converged.as_ref()),
        ))
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

/// The Fischer-Burmeister complementarity function.
///
/// ```math
/// \varphi(a, b) = a + b - \sqrt{a^2 + b^2}, \qquad \varphi(a,b) = 0 \iff a \geq 0,\ b \geq 0,\ ab = 0
/// ```
pub(crate) fn fischer_burmeister(a: Scalar, b: Scalar) -> Scalar {
    a + b - (a * a + b * b).sqrt()
}

/// The nested return-mapping solve: the plastic multiplier is converged at each load
/// step via [`ElasticPlastic::return_map`], with the analytically eliminated tangent
/// from [`ElasticPlastic::consistent_tangent_stiffness`] supplied to the solver.
fn nested_root<C: ElasticPlastic>(
    model: &C,
    applied_load: AppliedLoad,
    solver: impl FirstOrderRootFinding<
        FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness,
        DeformationGradient,
    >,
) -> Result<(Times, DeformationGradients, PlasticStateVariablesHistory), ConstitutiveError> {
    let (matrix, prescribed, time) = bcs(applied_load);
    let mut vector = Vector::zero(matrix.len());
    let mut state = model.initial_state();
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
                    let updated_state = model.return_map(deformation_gradient, &previous_state)?;
                    Ok(model
                        .first_piola_kirchhoff_stress(deformation_gradient, &updated_state.0)?)
                },
                |deformation_gradient: &DeformationGradient| {
                    Ok(model
                        .consistent_tangent_stiffness(deformation_gradient, &previous_state)?
                        .0)
                },
                deformation_gradient.clone(),
                EqualityConstraint::Linear(matrix.clone(), vector.clone()),
                None,
            )
            .map_err(|error| ConstitutiveError::upstream(error, model))?;
        state = model.return_map(&deformation_gradient, &previous_state)?;
        deformation_gradients.push(deformation_gradient.clone());
        states.push(state.clone());
    }
    Ok((
        time.iter().copied().collect(),
        deformation_gradients.into(),
        states.into(),
    ))
}

/// First-order root-finding methods for elastic-plastic solid constitutive models.
pub trait FirstOrderRoot {
    /// Solve for the unknown components of the deformation gradients under an applied load.
    ///
    /// With [`SolveStrategy::Condensed`], this is the nested return-mapping solve: the
    /// plastic step is converged at each outer iterate by [`ElasticPlastic::return_map`]
    /// and the analytically eliminated tangent from
    /// [`ElasticPlastic::consistent_tangent_stiffness`] is supplied to the solver. That is
    /// cheaper than the block solver, whose `K_uv`/`K_vu` assembly and per-step
    /// factorization are pure overhead on a system this small.
    ///
    /// [`SolveStrategy::Monolithic`] steps the deformation gradient and the coupled local
    /// unknowns $`(\mathbf{E},\Delta\gamma)`$ of the return map together through
    /// [`FirstOrderRootFindingBlock::root_block`], rather than converging the local
    /// block before every outer step. The local residual is the same coupled system the
    /// return map solves, with the yield inequality imposed by a Fischer-Burmeister
    /// complementarity residual so elastic steps recover $`\Delta\gamma = 0`$ on their
    /// own. It converges to the same step as [`SolveStrategy::Condensed`], also under
    /// non-proportional loading.
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderRootFindingBlock<
            DeformationGradient,
            Vector,
            FirstPiolaKirchhoffStress,
            Vector,
            FirstPiolaKirchhoffTangentStiffness,
            Matrix,
            Matrix,
            Matrix,
        > + FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
        strategy: SolveStrategy,
    ) -> Result<(Times, DeformationGradients, PlasticStateVariablesHistory), ConstitutiveError>;
}

impl<C> FirstOrderRoot for C
where
    C: ElasticPlastic,
{
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderRootFindingBlock<
            DeformationGradient,
            Vector,
            FirstPiolaKirchhoffStress,
            Vector,
            FirstPiolaKirchhoffTangentStiffness,
            Matrix,
            Matrix,
            Matrix,
        > + FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
        strategy: SolveStrategy,
    ) -> Result<(Times, DeformationGradients, PlasticStateVariablesHistory), ConstitutiveError>
    {
        if let SolveStrategy::Condensed(_) = strategy {
            return nested_root(self, applied_load, solver);
        }
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
        // Every local unknown is free, so there is nothing internal to pin: an empty
        // (zero-row) local constraint.
        let local_constraint = (
            CscMatrix::from_pattern(0, coupled::SIZE, Vec::new()),
            Vector::zero(0),
        );
        let mut state = self.initial_state();
        let mut deformation_gradient = DeformationGradient::identity();
        let mut deformation_gradients = vec![deformation_gradient.clone()];
        let mut states = vec![state.clone()];
        for time_step in time.iter().skip(1) {
            prescribed
                .iter()
                .for_each(|(index, function)| global_vector[*index] = function(*time_step));
            let previous_state = state.clone();
            let (deformation_gradient_new, local_new) = solver
                .root_block(
                    |global: &DeformationGradient, local: &Vector| {
                        coupled::monolithic_plastic(self, &previous_state, local)
                            .and_then(|plastic| self.first_piola_kirchhoff_stress(global, &plastic))
                            .map_err(|error| error.to_string())
                    },
                    |global: &DeformationGradient, local: &Vector| {
                        coupled::monolithic_residual_local(self, global, &previous_state, local)
                            .map_err(|error| error.to_string())
                    },
                    |global: &DeformationGradient, local: &Vector| {
                        coupled::monolithic_tangents(self, global, &previous_state, local)
                            .map_err(|error| error.to_string())
                    },
                    (deformation_gradient.clone(), Vector::zero(coupled::SIZE)),
                    (global_matrix.clone(), global_vector.clone()),
                    local_constraint.clone(),
                    None,
                    strategy.clone(),
                )
                .map_err(|error| ConstitutiveError::upstream(error, self))?;
            state = coupled::monolithic_state(self, &previous_state, &local_new)?;
            deformation_gradient = deformation_gradient_new;
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
