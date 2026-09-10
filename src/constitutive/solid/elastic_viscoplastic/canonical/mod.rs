#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        canonical::Canonical,
        fluid::{
            plastic::Plastic,
            viscoplastic::{Viscoplastic, ViscoplasticEvolution, ViscoplasticStateVariables},
        },
        solid::{
            elastic::Elastic,
            elastic_plastic::{Matrix3, entries_4, matrix_3, rank_2, rank_4},
            elastic_viscoplastic::{
                ElasticPlasticOrViscoplastic, ElasticViscoplastic, PlasticTangents,
            },
        },
    },
    math::{
        ContractFirstSecondWithSecond, ContractSecondWithFirst, ContractThirdWithFirst, Current,
        Derivative, Differentiate, Intermediate, Quantity, Rank2, Reference, Scalar, Tensor,
        TensorRank2, TensorRank4, TensorTuple,
        integrate::{ButcherTableau, Flat, IntegrableField, Product, StateEvolution, Unimodular},
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, CauchyTangentStiffnessElastic,
        CauchyTangentStiffnessPlastic, DeformationGradient, DeformationGradientPlastic,
        FirstPiolaKirchhoffStress, FirstPiolaKirchhoffStressElastic,
        FirstPiolaKirchhoffTangentStiffness, FirstPiolaKirchhoffTangentStiffnessElastic,
        FirstPiolaKirchhoffTangentStiffnessPlastic, MandelStressElastic,
        SecondPiolaKirchhoffStress, SecondPiolaKirchhoffStressElastic,
        SecondPiolaKirchhoffTangentStiffness, SecondPiolaKirchhoffTangentStiffnessElastic,
        StretchingRatePlastic,
    },
    units::{Dissipation, Rate, Stress, Time},
};
use std::{array::from_fn, ops::Add};

impl<C1, C2> Plastic for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Plastic,
{
    fn initial_yield_stress(&self) -> Quantity<Stress> {
        self.1.initial_yield_stress()
    }
    fn hardening_slope(&self) -> Quantity<Stress> {
        self.1.hardening_slope()
    }
}

impl<C1, C2, Y2> Viscoplastic<Y2> for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Viscoplastic<Y2>,
    Y2: Differentiate + Tensor,
{
    fn initial_state(&self) -> ViscoplasticStateVariables<Y2> {
        self.1.initial_state()
    }
    fn plastic_evolution(
        &self,
        mandel_stress: MandelStressElastic,
        state_variables: &ViscoplasticStateVariables<Y2>,
    ) -> Result<ViscoplasticEvolution<Y2>, ConstitutiveError> {
        self.1.plastic_evolution(mandel_stress, state_variables)
    }
    fn plastic_stretching_rate(
        &self,
        deviatoric_mandel_stress: MandelStressElastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<StretchingRatePlastic, ConstitutiveError> {
        self.1
            .plastic_stretching_rate(deviatoric_mandel_stress, yield_stress)
    }
    fn dissipation_potential(
        &self,
        plastic_stretching_rate: StretchingRatePlastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        self.1
            .dissipation_potential(plastic_stretching_rate, yield_stress)
    }
    fn dual_dissipation_potential(
        &self,
        deviatoric_mandel_stress: MandelStressElastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        self.1
            .dual_dissipation_potential(deviatoric_mandel_stress, yield_stress)
    }
    fn rate_sensitivity(&self) -> Scalar {
        self.1.rate_sensitivity()
    }
    fn reference_flow_rate(&self) -> Quantity<Rate> {
        self.1.reference_flow_rate()
    }
}

impl<C1, C2> ElasticPlasticOrViscoplastic for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Plastic,
{
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyStress, ConstitutiveError> {
        self.0
            .cauchy_stress(&(deformation_gradient * deformation_gradient_p.inverse()).into())
    }
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        Ok(
            CauchyTangentStiffnessElastic::from(self.0.cauchy_tangent_stiffness(
                &(deformation_gradient * &deformation_gradient_p_inverse).into(),
            )?) * deformation_gradient_p_inverse.transpose(),
        )
    }
    fn first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        Ok(
            FirstPiolaKirchhoffStressElastic::from(self.0.first_piola_kirchhoff_stress(
                &(deformation_gradient * &deformation_gradient_p_inverse).into(),
            )?) * deformation_gradient_p_inverse.transpose(),
        )
    }
    fn first_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        let deformation_gradient_p_inverse_transpose = deformation_gradient_p_inverse.transpose();
        Ok((FirstPiolaKirchhoffTangentStiffnessElastic::from(
            self.0.first_piola_kirchhoff_tangent_stiffness(
                &(deformation_gradient * &deformation_gradient_p_inverse).into(),
            )?,
        ) * &deformation_gradient_p_inverse_transpose)
            .contract_second_with_first(&deformation_gradient_p_inverse_transpose))
    }
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        Ok(&deformation_gradient_p_inverse
            * SecondPiolaKirchhoffStressElastic::from(self.0.second_piola_kirchhoff_stress(
                &(deformation_gradient * &deformation_gradient_p_inverse).into(),
            )?)
            * deformation_gradient_p_inverse.transpose())
    }
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        Ok((SecondPiolaKirchhoffTangentStiffnessElastic::from(
            self.0.second_piola_kirchhoff_tangent_stiffness(
                &(deformation_gradient * &deformation_gradient_p_inverse).into(),
            )?,
        ) * deformation_gradient_p_inverse.transpose())
        .contract_first_second_with_second(
            &deformation_gradient_p_inverse,
            &deformation_gradient_p_inverse,
        ))
    }
}

impl<C1, C2> PlasticTangents for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Plastic,
{
    fn cauchy_tangent_stiffness_p(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyTangentStiffnessPlastic, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        let deformation_gradient_e = deformation_gradient * &deformation_gradient_p_inverse;
        Ok(CauchyTangentStiffnessElastic::from(
            self.0
                .cauchy_tangent_stiffness(&deformation_gradient_e.clone().into())?,
        )
        .contract_third_with_first(&deformation_gradient_e)
            * deformation_gradient_p_inverse.transpose()
            * -1.0)
    }
    fn first_piola_kirchhoff_tangent_stiffness_p(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffTangentStiffnessPlastic, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        let deformation_gradient_p_inverse_transpose = deformation_gradient_p_inverse.transpose();
        let deformation_gradient_e = deformation_gradient * &deformation_gradient_p_inverse;
        let first_piola_kirchhoff_stress =
            self.first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?;
        Ok(((FirstPiolaKirchhoffTangentStiffnessElastic::from(
            self.0
                .first_piola_kirchhoff_tangent_stiffness(&deformation_gradient_e.clone().into())?,
        )
        .contract_third_with_first(&deformation_gradient_e)
            * &deformation_gradient_p_inverse_transpose)
            .contract_second_with_first(&deformation_gradient_p_inverse_transpose)
            + FirstPiolaKirchhoffTangentStiffnessPlastic::dyad_il_kj(
                &first_piola_kirchhoff_stress,
                &deformation_gradient_p_inverse_transpose,
            ))
            * -1.0)
    }
}

impl<C1, C2, Y2> ElasticViscoplastic<Y2> for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Viscoplastic<Y2>,
    Y2: Differentiate + Tensor,
{
}

/// The internal state `(F_p, Y)` evolves as `F_p` on the unimodular group
/// (`Reference → Intermediate`, so its algebra element `D_p Δt` is
/// `Intermediate → Intermediate`) and the hardening variable `Y` additively. The
/// rate is `(D_p, Ẏ)`, from the model's [`plastic_evolution`] (`D_p` recovered as
/// `Ḟ_p F_p⁻¹`), driven by the total deformation gradient through the Mandel
/// stress.
///
/// [`plastic_evolution`]: Viscoplastic::plastic_evolution
impl<C1, C2, Y> StateEvolution<Time, Y> for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Viscoplastic<Y>,
    Y: Clone + Differentiate<Time> + Tensor,
    for<'a> Y: Add<&'a Y, Output = Y>,
    TensorTuple<TensorRank2<3, Intermediate, Intermediate>, Y>: Differentiate<
            Time,
            Derivative = TensorTuple<
                TensorRank2<3, Intermediate, Intermediate, Rate>,
                Derivative<Y>,
            >,
        >,
{
    type Field = Product<Unimodular<Intermediate, Reference>, Flat<Y>>;
    type Drive = DeformationGradient;
    fn initial_state(&self) -> ViscoplasticStateVariables<Y> {
        <Self as Viscoplastic<Y>>::initial_state(self)
    }
    fn state_rate(
        &self,
        _time: Quantity<Time>,
        deformation_gradient: &DeformationGradient,
        state: &ViscoplasticStateVariables<Y>,
    ) -> Result<Derivative<<Self::Field as IntegrableField>::Increment, Time>, String> {
        let mandel_stress = self.mandel_stress(deformation_gradient, &state.0)?;
        let evolution = self.plastic_evolution(mandel_stress, state)?;
        let plastic_stretching_rate = evolution.0 * state.0.inverse();
        Ok(TensorTuple(plastic_stretching_rate, evolution.1))
    }
}

type AlgebraElement = TensorRank2<3, Intermediate, Intermediate>;
type Directions = [Matrix3; 9];

const ZERO_3: Matrix3 = [[0.0; 3]; 3];
const ZERO_9: Directions = [ZERO_3; 9];

fn multiply(a: &Matrix3, b: &Matrix3) -> Matrix3 {
    from_fn(|i| from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn add_scaled(target: &mut Matrix3, factor: Scalar, source: &Matrix3) {
    (0..3).for_each(|i| (0..3).for_each(|j| target[i][j] += factor * source[i][j]))
}

fn scaled(source: &Matrix3, factor: Scalar) -> Matrix3 {
    from_fn(|i| from_fn(|j| factor * source[i][j]))
}

/// One RKMK step of the plastic state together with its algorithmic tangents
/// with respect to the deformation gradient held fixed across the step.
pub struct RkmkStepTangent {
    /// The advanced state.
    pub state: ViscoplasticStateVariables<Quantity>,
    /// The tangent of the plastic deformation gradient.
    pub deformation_gradient_p_tangent: TensorRank4<3, Intermediate, Reference, Current, Reference>,
    /// The tangent of the hardening variable.
    pub hardening_tangent: TensorRank2<3, Current, Reference>,
}

impl<C1, C2> Canonical<C1, C2>
where
    C1: Elastic,
    C2: Viscoplastic<Quantity>,
{
    //
    // D_p = D_p(dev M(F, F_p), Y(S)), so a direction in F moves it through
    // dM/dF, through dM/dF_p seen by the carried dF_p/dF, and through the
    // hardening slope seen by the carried dS/dF. The equivalent rate |D_p|
    // differentiates as the flow direction contracted with dD_p.
    //
    #[allow(clippy::type_complexity)]
    fn plastic_rate_and_tangent(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &Matrix3,
        hardening: Scalar,
        deformation_gradient_p_tangent: &Directions,
        hardening_tangent: &[Scalar; 9],
    ) -> Result<(Matrix3, Scalar, Directions, [Scalar; 9]), ConstitutiveError> {
        let plastic: DeformationGradientPlastic = rank_2(deformation_gradient_p);
        let mandel_stress = self.mandel_stress(deformation_gradient, &plastic)?;
        let deviatoric = mandel_stress.deviatoric();
        let yield_stress = self.yield_stress(Quantity::new(hardening))?;
        let plastic_stretching_rate = <Self as Viscoplastic<Quantity>>::plastic_stretching_rate(
            self,
            deviatoric.clone(),
            yield_stress,
        )?;
        let magnitude = plastic_stretching_rate.norm().value();
        let rate = matrix_3(&plastic_stretching_rate);
        let flow_direction = if magnitude > 0.0 {
            scaled(&rate, 1.0 / magnitude)
        } else {
            ZERO_3
        };
        let mandel_tangent =
            entries_4(&self.mandel_stress_tangent(deformation_gradient, &plastic)?);
        let mandel_tangent_p =
            entries_4(&self.mandel_stress_tangent_p(deformation_gradient, &plastic)?);
        let rate_tangent = entries_4(
            &<Self as Viscoplastic<Quantity>>::plastic_stretching_rate_tangent(
                self,
                &deviatoric,
                yield_stress,
            )?,
        );
        let rate_tangent_yield = matrix_3(
            &<Self as Viscoplastic<Quantity>>::plastic_stretching_rate_tangent_yield(
                self,
                deviatoric,
                yield_stress,
            )?,
        );
        let hardening_slope = self.hardening_slope().value();
        let mut rate_directions = ZERO_9;
        let mut equivalent_directions = [0.0; 9];
        for (direction, (rate_direction, equivalent_direction)) in rate_directions
            .iter_mut()
            .zip(equivalent_directions.iter_mut())
            .enumerate()
        {
            let (k, l) = (direction / 3, direction % 3);
            let mut mandel: Matrix3 = from_fn(|i| {
                from_fn(|j| {
                    mandel_tangent[i][j][k][l]
                        + (0..3)
                            .map(|n| {
                                (0..3)
                                    .map(|o| {
                                        mandel_tangent_p[i][j][n][o]
                                            * deformation_gradient_p_tangent[direction][n][o]
                                    })
                                    .sum::<Scalar>()
                            })
                            .sum::<Scalar>()
                })
            });
            let trace = (mandel[0][0] + mandel[1][1] + mandel[2][2]) / 3.0;
            (0..3).for_each(|i| mandel[i][i] -= trace);
            *rate_direction = from_fn(|i| {
                from_fn(|j| {
                    (0..3)
                        .map(|a| {
                            (0..3)
                                .map(|b| rate_tangent[i][j][a][b] * mandel[a][b])
                                .sum::<Scalar>()
                        })
                        .sum::<Scalar>()
                        + rate_tangent_yield[i][j] * hardening_slope * hardening_tangent[direction]
                })
            });
            *equivalent_direction = (0..3)
                .map(|i| {
                    (0..3)
                        .map(|j| flow_direction[i][j] * rate_direction[i][j])
                        .sum::<Scalar>()
                })
                .sum();
        }
        Ok((rate, magnitude, rate_directions, equivalent_directions))
    }
    //
    // exp(sigma) F_p, and its directions through the Frechet derivative of the
    // exponential contracted with the carried directions of sigma.
    //
    fn exponential_action(
        &self,
        sigma: &Matrix3,
        sigma_tangent: &Directions,
        base: &Matrix3,
    ) -> Result<(Matrix3, Directions), ConstitutiveError> {
        let algebra: AlgebraElement = rank_2(sigma);
        let exponential = matrix_3(
            &algebra
                .expm()
                .map_err(|error| ConstitutiveError::upstream(error, self))?,
        );
        let exponential_tangent = entries_4(
            &algebra
                .dexpm()
                .map_err(|error| ConstitutiveError::upstream(error, self))?,
        );
        let mut directions = ZERO_9;
        for (direction, entry) in directions.iter_mut().enumerate() {
            let derivative: Matrix3 = from_fn(|a| {
                from_fn(|c| {
                    (0..3)
                        .map(|p| {
                            (0..3)
                                .map(|q| {
                                    exponential_tangent[a][c][p][q] * sigma_tangent[direction][p][q]
                                })
                                .sum::<Scalar>()
                        })
                        .sum::<Scalar>()
                })
            });
            *entry = multiply(&derivative, base)
        }
        Ok((multiply(&exponential, base), directions))
    }
    /// Advances `state` one fixed `Tab`-tableau RKMK step with the deformation
    /// gradient frozen, carrying the algorithmic tangents `dF_p/dF` and `dY/dF`
    /// forward through the stage sweep.
    ///
    /// The step is `F_p^{n+1} = exp(σ) F_p^n` with `σ = Σᵢ bᵢ k̃ᵢ`,
    /// `k̃ᵢ = dexpinv_{σᵢ}(fᵢ Δt)` and `σᵢ = Σ_{j<i} aᵢⱼ k̃ⱼ`, so a direction in
    /// `F` propagates through [`TensorRank2::dexpm`] at each stage point, the
    /// rate linearization, and [`TensorRank2::dexpinv_tangent`].
    pub fn rkmk_step_tangent<Tab>(
        &self,
        deformation_gradient: &DeformationGradient,
        state: &ViscoplasticStateVariables<Quantity>,
        time_step: Quantity<Time>,
    ) -> Result<RkmkStepTangent, ConstitutiveError>
    where
        Tab: ButcherTableau,
    {
        let initial = matrix_3(&state.0);
        let initial_hardening = state.1.value();
        let step = time_step.value();
        let mut slopes: Vec<(Matrix3, Scalar)> = Vec::with_capacity(Tab::STAGES);
        let mut slope_tangents: Vec<(Directions, [Scalar; 9])> = Vec::with_capacity(Tab::STAGES);
        for i in 0..Tab::STAGES {
            let mut sigma = ZERO_3;
            let mut sigma_hardening = 0.0;
            let mut sigma_tangent = ZERO_9;
            let mut sigma_hardening_tangent = [0.0; 9];
            for j in 0..i {
                let weight = Tab::A[i][j];
                add_scaled(&mut sigma, weight, &slopes[j].0);
                sigma_hardening += weight * slopes[j].1;
                for direction in 0..9 {
                    add_scaled(
                        &mut sigma_tangent[direction],
                        weight,
                        &slope_tangents[j].0[direction],
                    );
                    sigma_hardening_tangent[direction] += weight * slope_tangents[j].1[direction];
                }
            }
            let (point, point_tangent) = if i == 0 {
                (initial, ZERO_9)
            } else {
                self.exponential_action(&sigma, &sigma_tangent, &initial)?
            };
            let (rate, equivalent_rate, rate_tangent, equivalent_rate_tangent) = self
                .plastic_rate_and_tangent(
                    deformation_gradient,
                    &point,
                    initial_hardening + sigma_hardening,
                    &point_tangent,
                    &sigma_hardening_tangent,
                )?;
            let increment = scaled(&rate, step);
            let increment_tangent: Directions =
                from_fn(|direction| scaled(&rate_tangent[direction], step));
            let increment_hardening_tangent: [Scalar; 9] =
                from_fn(|direction| equivalent_rate_tangent[direction] * step);
            if i == 0 {
                slopes.push((increment, equivalent_rate * step));
                slope_tangents.push((increment_tangent, increment_hardening_tangent));
            } else {
                let algebra: AlgebraElement = rank_2(&sigma);
                let rate_element: AlgebraElement = rank_2(&increment);
                let mut slope_tangent = ZERO_9;
                for (direction, entry) in slope_tangent.iter_mut().enumerate() {
                    *entry = matrix_3(&algebra.dexpinv_tangent(
                        &rate_element,
                        &rank_2(&sigma_tangent[direction]),
                        &rank_2(&increment_tangent[direction]),
                    ))
                }
                slopes.push((
                    matrix_3(&algebra.dexpinv(&rate_element)),
                    equivalent_rate * step,
                ));
                slope_tangents.push((slope_tangent, increment_hardening_tangent));
            }
        }
        let mut sigma = ZERO_3;
        let mut sigma_hardening = 0.0;
        let mut sigma_tangent = ZERO_9;
        let mut sigma_hardening_tangent = [0.0; 9];
        for (i, weight) in Tab::B.iter().enumerate().take(Tab::STAGES) {
            add_scaled(&mut sigma, *weight, &slopes[i].0);
            sigma_hardening += weight * slopes[i].1;
            for direction in 0..9 {
                add_scaled(
                    &mut sigma_tangent[direction],
                    *weight,
                    &slope_tangents[i].0[direction],
                );
                sigma_hardening_tangent[direction] += weight * slope_tangents[i].1[direction];
            }
        }
        let (point, point_tangent) = self.exponential_action(&sigma, &sigma_tangent, &initial)?;
        Ok(RkmkStepTangent {
            state: TensorTuple(
                rank_2(&point),
                Quantity::new(initial_hardening + sigma_hardening),
            ),
            deformation_gradient_p_tangent: rank_4(&from_fn(|a| {
                from_fn(|b| from_fn(|k| from_fn(|l| point_tangent[3 * k + l][a][b])))
            })),
            hardening_tangent: rank_2(&from_fn(|k| {
                from_fn(|l| sigma_hardening_tangent[3 * k + l])
            })),
        })
    }
}
