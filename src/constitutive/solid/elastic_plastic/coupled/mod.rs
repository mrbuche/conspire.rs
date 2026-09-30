#[cfg(test)]
mod test;

use super::{ElasticPlastic, fischer_burmeister};
use crate::{
    constitutive::{ConstitutiveError, fluid::plastic::PlasticStateVariables},
    math::{
        ContractSecondWithFirst, ContractThirdFourthWithFirstSecond, ContractWith, Intermediate,
        Matrix, Quantity, Rank2, Reference, SquareMatrix, Tensor, TensorArray, TensorRank2,
        TensorRank4, Vector,
        optimize::{NewtonRaphson, converged, limit_decrement},
    },
    mechanics::{
        DeformationGradient, DeformationGradientElastic, DeformationGradientGeneral,
        DeformationGradientPlastic, FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness,
        FlowDirectionPlastic, MandelStressElastic, Scalar,
    },
    units::Stress,
};
use std::{array::from_fn, fmt::Debug};

pub(crate) const SIZE: usize = 10;
const INITIAL_MULTIPLIER: Scalar = 1e-3;

type Unknowns = [Scalar; SIZE];

fn basis<I, J>(a: usize, b: usize) -> TensorRank2<3, I, J> {
    from_fn::<_, 3, _>(|i| from_fn::<_, 3, _>(|j| if i == a && j == b { 1.0 } else { 0.0 })).into()
}

fn failure<C: ElasticPlastic>(model: &C, error: &dyn Debug) -> ConstitutiveError {
    ConstitutiveError::custom(format!("{error:?}"), model)
}

fn increment(x: &Unknowns) -> FlowDirectionPlastic {
    from_fn(|i| from_fn::<_, 3, _>(|j| x[3 * i + j])).into()
}

/// The elastic-plastic stress and its tangent at fixed plastic deformation gradient,
/// from which the derivatives of the first Piola-Kirchhoff and Mandel stresses along
/// any $`(\mathrm{d}\mathbf{F},\mathrm{d}\mathbf{F}_\mathrm{p})`$ follow.
///
/// The model's Mandel stress carries $`\det\mathbf{F}`$, whereas
/// $`\mathbf{F}_\mathrm{e}^T\mathbf{P}\mathbf{F}_\mathrm{p}^T`$ carries
/// $`\det\mathbf{F}_\mathrm{e}`$; the two differ by $`\det\mathbf{F}_\mathrm{p}`$,
/// which the last step of `mandel_derivative` accounts for.
struct Linearization {
    tangent: FirstPiolaKirchhoffTangentStiffness,
    stress: FirstPiolaKirchhoffStress,
    f_p: DeformationGradientPlastic,
    f_p_inverse: DeformationGradientGeneral<Reference, Intermediate>,
    f_e: DeformationGradientElastic,
    mandel: MandelStressElastic,
}

impl Linearization {
    fn new<C: ElasticPlastic>(
        model: &C,
        f: &DeformationGradient,
        f_p: &DeformationGradientPlastic,
    ) -> Result<Self, ConstitutiveError> {
        let f_p_inverse = f_p.inverse();
        let stress = model.first_piola_kirchhoff_stress(f, f_p)?;
        let f_e = f * &f_p_inverse;
        let mandel = f_e.transpose() * &stress * f_p.transpose();
        Ok(Self {
            tangent: model.first_piola_kirchhoff_tangent_stiffness(f, f_p)?,
            stress,
            f_p: f_p.clone(),
            f_p_inverse,
            f_e,
            mandel,
        })
    }

    fn stress_derivative(
        &self,
        d_f: &DeformationGradient,
        d_f_p: &DeformationGradientPlastic,
    ) -> FirstPiolaKirchhoffStress {
        let by_f = (&self.tangent).contract_third_fourth_with_first_second(d_f);
        let by_f_p = (&self.tangent).contract_third_fourth_with_first_second(&(&self.f_e * d_f_p));
        let correction = &self.stress * d_f_p.transpose() * self.f_p_inverse.transpose();
        by_f - by_f_p - correction
    }

    fn mandel_derivative(
        &self,
        d_f: &DeformationGradient,
        d_f_p: &DeformationGradientPlastic,
    ) -> MandelStressElastic {
        let d_f_e = d_f * &self.f_p_inverse - &self.f_e * d_f_p * &self.f_p_inverse;
        let d_p = self.stress_derivative(d_f, d_f_p);
        let (f_e_t, f_p_t) = (self.f_e.transpose(), self.f_p.transpose());
        let d_m = d_f_e.transpose() * &self.stress * &f_p_t
            + &f_e_t * &d_p * &f_p_t
            + &f_e_t * &self.stress * d_f_p.transpose();
        let d_m = d_m + &self.mandel * (&self.f_p_inverse * d_f_p).trace();
        d_m * self.f_p.determinant()
    }
}

/// Everything evaluated once per iterate: the plastic deformation gradient, the
/// deviatoric Mandel stress with its equivalent stress and flow direction, the residual,
/// and the multiplier with the hardening modulus at its plastic strain, which the
/// Jacobian needs.
struct Iterate {
    plastic: DeformationGradientPlastic,
    deviatoric: MandelStressElastic,
    unit: FlowDirectionPlastic,
    direction: FlowDirectionPlastic,
    magnitude: Quantity<Stress>,
    residual: Unknowns,
    gamma: Scalar,
    hardening_modulus: Scalar,
}

impl Iterate {
    fn new<C: ElasticPlastic>(
        model: &C,
        f: &DeformationGradient,
        f_p_n: &DeformationGradientPlastic,
        strain_n: Scalar,
        x: &Unknowns,
    ) -> Result<Self, ConstitutiveError> {
        let plastic = increment(x)
            .expm()
            .map_err(|error| failure(model, &error))?
            * f_p_n;
        let deviatoric: MandelStressElastic = model.mandel_stress(f, &plastic)?.deviatoric();
        let magnitude = model.equivalent_stress(&deviatoric)?;
        let unit = model.flow_direction(&deviatoric)?;
        let direction = unit.symmetric_part();
        let mut residual = [0.0; SIZE];
        let flow = increment(x) - &direction * x[9];
        (0..3).for_each(|i| (0..3).for_each(|j| residual[3 * i + j] = flow[i][j].value()));
        residual[9] = model
            .yield_function(&deviatoric, Quantity::new(strain_n + x[9]))?
            .value();
        Ok(Self {
            plastic,
            deviatoric,
            unit,
            direction,
            magnitude,
            residual,
            gamma: x[9],
            hardening_modulus: model
                .hardening_modulus(Quantity::new(strain_n + x[9]))?
                .value(),
        })
    }
}

/// The linearization at an iterate together with the slopes
/// $`\partial\mathbf{F}_\mathrm{p}/\partial E_{ab}`$ of the exponential map.
struct Sensitivities<'a, C> {
    model: &'a C,
    linearization: Linearization,
    iterate: &'a Iterate,
    slopes: TensorRank4<3, Intermediate, Reference, Intermediate, Intermediate>,
}

impl<'a, C: ElasticPlastic> Sensitivities<'a, C> {
    fn new(
        model: &'a C,
        f: &DeformationGradient,
        f_p_n: &DeformationGradientPlastic,
        x: &Unknowns,
        iterate: &'a Iterate,
    ) -> Result<Self, ConstitutiveError> {
        let slopes = increment(x)
            .dexpm()
            .map_err(|error| failure(model, &error))?
            .contract_second_with_first(f_p_n);
        Ok(Self {
            model,
            linearization: Linearization::new(model, f, &iterate.plastic)?,
            iterate,
            slopes,
        })
    }

    /// The slope of the plastic deformation gradient along the direction $`E_{ab}`$.
    fn slope(&self, a: usize, b: usize) -> DeformationGradientPlastic {
        (&self.slopes).contract_third_fourth_with_first_second(&basis(a, b))
    }

    /// The slope of the symmetrized flow direction and of the equivalent stress along a
    /// Mandel stress increment.
    fn direction_slope(
        &self,
        d_m: &MandelStressElastic,
    ) -> Result<(FlowDirectionPlastic, Scalar), ConstitutiveError> {
        let Iterate {
            unit,
            magnitude,
            deviatoric,
            ..
        } = self.iterate;
        // the equivalent stress is not differentiable where the deviator vanishes, and
        // the flow direction is zero there: the trial state is elastic
        if magnitude.is_zero() {
            return Ok((FlowDirectionPlastic::zero(), 0.0));
        }
        let increment = d_m.deviatoric();
        let d_magnitude = increment.contract_with(unit).value();
        let d_unit = self.model.flow_direction_slope(deviatoric, &increment)?;
        Ok((d_unit.symmetric_part(), d_magnitude))
    }

    /// The Jacobian of the residual with respect to $`(\mathbf{E},\Delta\gamma)`$.
    fn jacobian(&self) -> Result<[[Scalar; SIZE]; SIZE], ConstitutiveError> {
        let Iterate {
            gamma,
            hardening_modulus,
            ..
        } = self.iterate;
        let mut jacobian = [[0.0; SIZE]; SIZE];
        for a in 0..3 {
            for b in 0..3 {
                let d_m = self
                    .linearization
                    .mandel_derivative(&DeformationGradient::zero(), &self.slope(a, b));
                let (d_direction, d_magnitude) = self.direction_slope(&d_m)?;
                for i in 0..3 {
                    for j in 0..3 {
                        let identity = if i == a && j == b { 1.0 } else { 0.0 };
                        jacobian[3 * i + j][3 * a + b] =
                            identity - gamma * d_direction[i][j].value();
                    }
                }
                jacobian[9][3 * a + b] = d_magnitude;
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                jacobian[3 * i + j][9] = -self.iterate.direction[i][j].value();
            }
        }
        jacobian[9][9] = -*hardening_modulus;
        Ok(jacobian)
    }
}

/// A converged step: the unknowns and the state they were evaluated at.
pub(super) struct Converged {
    x: Unknowns,
    iterate: Iterate,
}

/// Coupled Newton solve of the fully implicit plastic step.
///
/// The unknowns are the plastic increment $`\mathbf{E} = \Delta\gamma\,\mathbf{N}`$,
/// carried as nine components so that symmetry and tracelessness are enforced by the
/// residual rather than by a basis, and $`\Delta\gamma`$. The residuals are
/// ```math
/// \mathbf{E} - \Delta\gamma\,\mathbf{N}\big(\mathbf{M}'(\mathbf{F},\exp(\mathbf{E})\mathbf{F}_\mathrm{p}^n)\big) = \mathbf{0},
/// \qquad f\big(\mathbf{M}',\varepsilon_\mathrm{p}^n+\Delta\gamma\big) = 0,
/// ```
/// so the flow direction is the end-of-step one. The yield equation is solved as the
/// Fischer-Burmeister function of the monolithic system, which has the same root for a
/// plastic step and the same residual, so this is the nested local solve of
/// [`return_map`](super::ElasticPlastic::return_map) and [`condensed`]. The monolithic
/// strategies do not call it: they step the deformation gradient and these unknowns
/// together. It is converged and limited as `local_solver` says. Returns `None` for an
/// elastic step.
pub(super) fn solve<C: ElasticPlastic>(
    model: &C,
    f: &DeformationGradient,
    state: &PlasticStateVariables,
    local_solver: &NewtonRaphson,
) -> Result<Option<Converged>, ConstitutiveError> {
    let (f_p_n, &strain_n): (&DeformationGradientPlastic, &Quantity) = state.into();
    let strain_n = strain_n.value();
    let deviatoric = model.mandel_stress(f, f_p_n)?.deviatoric();
    if model
        .yield_function(&deviatoric, Quantity::new(strain_n))?
        .value()
        <= 0.0
    {
        return Ok(None);
    }
    let direction = model.flow_direction(&deviatoric)?.symmetric_part();
    let reference = reference(model);
    let mut x = [0.0; SIZE];
    (0..3).for_each(|i| {
        (0..3).for_each(|j| x[3 * i + j] = INITIAL_MULTIPLIER * direction[i][j].value())
    });
    x[SIZE - 1] = INITIAL_MULTIPLIER;
    let mut iterate = Iterate::new(model, f, f_p_n, strain_n, &x)?;
    let mut scales = None;
    let mut steps = 0;
    loop {
        let residual = monolithic_local_residual(&iterate, x[SIZE - 1], reference);
        if converged(local_solver, &residual, SIZE, &mut scales) {
            return Ok(Some(Converged { x, iterate }));
        } else if steps == local_solver.max_steps {
            return Err(failure(
                model,
                &format!(
                    "The coupled return mapping did not converge in {} steps.",
                    local_solver.max_steps
                ),
            ));
        }
        steps += 1;
        let (k_vv, _) = local_jacobian(
            &Sensitivities::new(model, f, f_p_n, &x, &iterate)?.jacobian()?,
            x[SIZE - 1],
            -iterate.residual[SIZE - 1] / reference,
            reference,
        );
        let mut decrement = k_vv
            .into_iter()
            .collect::<SquareMatrix>()
            .solve_lu(&residual)
            .map_err(|error| failure(model, &error))?;
        limit_decrement(local_solver, &mut [(&mut decrement, SIZE)]);
        (0..SIZE).for_each(|i| x[i] -= decrement[i]);
        iterate = Iterate::new(model, f, f_p_n, strain_n, &x)?;
    }
}

/// The scale of the yield row of the local residual, the initial yield stress, or unity
/// where a perfectly weak material has none.
fn reference<C: ElasticPlastic>(model: &C) -> Scalar {
    let initial = model.initial_yield_stress().value();
    if initial > 0.0 { initial } else { 1.0 }
}

pub(super) fn updated_state(
    state: &PlasticStateVariables,
    converged: Option<&Converged>,
) -> PlasticStateVariables {
    match converged {
        None => state.clone(),
        Some(Converged { x, iterate }) => {
            let (_, &strain_n): (&DeformationGradientPlastic, &Quantity) = state.into();
            (iterate.plastic.clone(), strain_n + Quantity::new(x[9])).into()
        }
    }
}

/// The plastic deformation gradient of a monolithic trial state: the local unknowns
/// are $`(\mathbf{E},\Delta\gamma)`$ and $`\mathbf{F}_\mathrm{p}=\exp(\mathbf{E})\mathbf{F}_\mathrm{p}^n`$.
pub(super) fn monolithic_plastic<C: ElasticPlastic>(
    model: &C,
    state: &PlasticStateVariables,
    local: &Vector,
) -> Result<DeformationGradientPlastic, ConstitutiveError> {
    let (f_p_n, _): (&DeformationGradientPlastic, &Quantity) = state.into();
    let x: Unknowns = from_fn(|i| local[i]);
    Ok(increment(&x)
        .expm()
        .map_err(|error| failure(model, &error))?
        * f_p_n)
}

/// The plastic state a monolithic solve arrives at.
pub(crate) fn monolithic_state<C: ElasticPlastic>(
    model: &C,
    state: &PlasticStateVariables,
    local: &Vector,
) -> Result<PlasticStateVariables, ConstitutiveError> {
    let (_, &strain_n): (&DeformationGradientPlastic, &Quantity) = state.into();
    Ok((
        monolithic_plastic(model, state, local)?,
        strain_n + Quantity::new(local[SIZE - 1]),
    )
        .into())
}

/// The local residual of the monolithic system: the flow rule as in [`solve`], and the
/// yield inequality imposed as the Fischer-Burmeister complementarity
/// $`\varphi(\Delta\gamma,-f/Y_0)`$, so an elastic step recovers $`\Delta\gamma=0`$ on
/// its own.
fn monolithic_local_residual(iterate: &Iterate, gamma: Scalar, reference: Scalar) -> Vector {
    let mut residual = Vector::zero(SIZE);
    (0..SIZE - 1).for_each(|row| residual[row] = iterate.residual[row]);
    residual[SIZE - 1] = fischer_burmeister(gamma, -iterate.residual[SIZE - 1] / reference);
    residual
}

pub(super) fn monolithic_residual_local<C: ElasticPlastic>(
    model: &C,
    f: &DeformationGradient,
    state: &PlasticStateVariables,
    local: &Vector,
) -> Result<Vector, ConstitutiveError> {
    let (f_p_n, &strain_n): (&DeformationGradientPlastic, &Quantity) = state.into();
    let x: Unknowns = from_fn(|i| local[i]);
    let iterate = Iterate::new(model, f, f_p_n, strain_n.value(), &x)?;
    Ok(monolithic_local_residual(
        &iterate,
        x[SIZE - 1],
        reference(model),
    ))
}

/// The tangent blocks $`(K_{uu},K_{vu},K_{uv},K_{vv})`$ of the monolithic system in the
/// order the block solver takes them, with the global unknown $`\mathbf{F}`$ and the
/// local unknowns $`(\mathbf{E},\Delta\gamma)`$.
///
/// The local residual is the coupled one of [`solve`] with its yield row replaced by
/// the Fischer-Burmeister function, so those blocks are the coupled Jacobian's with that
/// row scaled by $`\partial\varphi/\partial b\,(-1/Y_0)`$ and the multiplier column
/// carrying the extra $`\partial\varphi/\partial a`$. $`K_{uu}`$ is the continuum
/// tangent, since the flow direction depends on $`\mathbf{F}`$ only through the local
/// unknowns.
pub(super) fn monolithic_tangents<C: ElasticPlastic>(
    model: &C,
    f: &DeformationGradient,
    state: &PlasticStateVariables,
    local: &Vector,
) -> Result<(FirstPiolaKirchhoffTangentStiffness, Matrix, Matrix, Matrix), ConstitutiveError> {
    let Monolithic {
        tangent_uu,
        tangent_vu,
        tangent_uv,
        tangent_vv,
        ..
    } = monolithic_evaluate(model, f, state, local)?;
    Ok((tangent_uu, tangent_vu, tangent_uv, tangent_vv))
}

/// The local block $`K_{vv}`$ of the monolithic system from the coupled Jacobian: its
/// yield row is replaced by the Fischer-Burmeister function of
/// $`(a,b)=(\Delta\gamma,-f/Y_0)`$. Also returns the factor that row was scaled by.
fn local_jacobian(
    jacobian: &[[Scalar; SIZE]; SIZE],
    a: Scalar,
    b: Scalar,
    reference: Scalar,
) -> (Matrix, Scalar) {
    let radius = (a * a + b * b).sqrt();
    let (partial_a, partial_b) = if radius > 0.0 {
        (1.0 - a / radius, 1.0 - b / radius)
    } else {
        (1.0, 1.0)
    };
    let factor = -partial_b / reference;
    let mut k_vv = Matrix::zero(SIZE, SIZE);
    for row in 0..SIZE - 1 {
        for column in 0..SIZE {
            k_vv[row][column] = jacobian[row][column];
        }
    }
    for column in 0..SIZE - 1 {
        k_vv[SIZE - 1][column] = factor * jacobian[SIZE - 1][column];
    }
    k_vv[SIZE - 1][SIZE - 1] = partial_a + factor * jacobian[SIZE - 1][SIZE - 1];
    (k_vv, factor)
}

/// The stress, the consistent tangent and the updated plastic state of one step, from a
/// local solve of the monolithic system's unknowns $`(\mathbf{E},\Delta\gamma)`$ alone
/// followed by the Schur complement that eliminates them,
/// $`\mathcal{C}_\mathrm{eff} = K_{uu} - K_{uv}K_{vv}^{-1}K_{vu}`$.
///
/// This is the condensed strategy of the block solver at one integration point, with
/// no state carried between calls, the local solve converged and limited as the local
/// solver says. An elastic step has nothing to solve for.
pub(crate) fn condensed<C: ElasticPlastic>(
    model: &C,
    f: &DeformationGradient,
    state: &PlasticStateVariables,
    local_solver: &NewtonRaphson,
) -> Result<
    (
        FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness,
        PlasticStateVariables,
    ),
    ConstitutiveError,
> {
    let (f_p_n, _): (&DeformationGradientPlastic, &Quantity) = state.into();
    let Some(Converged { x, iterate }) = solve(model, f, state, local_solver)? else {
        return Ok((
            model.first_piola_kirchhoff_stress(f, f_p_n)?,
            model.first_piola_kirchhoff_tangent_stiffness(f, f_p_n)?,
            state.clone(),
        ));
    };
    let Monolithic {
        stress,
        tangent_uu: tangent,
        tangent_vu: k_vu,
        tangent_uv: k_uv,
        tangent_vv: k_vv,
        ..
    } = evaluate(model, f, f_p_n, &x, &iterate)?;
    let lu = k_vv
        .into_iter()
        .collect::<SquareMatrix>()
        .factorize_lu()
        .map_err(|error| failure(model, &error))?;
    let solved: Vec<Vector> = (0..9)
        .map(|column| {
            lu.solve(&Vector::from(
                (0..SIZE).map(|row| k_vu[row][column]).collect::<Vec<_>>(),
            ))
        })
        .collect();
    let mut effective = tangent;
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    effective[i][j][k][l] -= Quantity::new(
                        (0..SIZE)
                            .map(|m| k_uv[3 * i + j][m] * solved[3 * k + l][m])
                            .sum::<Scalar>(),
                    )
                }
            }
        }
    }
    Ok((
        stress,
        effective,
        updated_state(state, Some(&Converged { x, iterate })),
    ))
}

/// Everything the monolithic system needs at a point from one evaluation: the first
/// Piola-Kirchhoff stress at the trial plastic state, the local residual, and the
/// tangent blocks $`(K_{uu},K_{vu},K_{uv},K_{vv})`$ of [`monolithic_tangents`].
pub(crate) struct Monolithic {
    pub(crate) stress: FirstPiolaKirchhoffStress,
    #[cfg_attr(not(feature = "fem"), allow(dead_code))]
    pub(crate) residual_local: Vector,
    pub(crate) tangent_uu: FirstPiolaKirchhoffTangentStiffness,
    pub(crate) tangent_vu: Matrix,
    pub(crate) tangent_uv: Matrix,
    pub(crate) tangent_vv: Matrix,
}

pub(crate) fn monolithic_evaluate<C: ElasticPlastic>(
    model: &C,
    f: &DeformationGradient,
    state: &PlasticStateVariables,
    local: &Vector,
) -> Result<Monolithic, ConstitutiveError> {
    let (f_p_n, &strain_n): (&DeformationGradientPlastic, &Quantity) = state.into();
    let x: Unknowns = from_fn(|i| local[i]);
    let iterate = Iterate::new(model, f, f_p_n, strain_n.value(), &x)?;
    evaluate(model, f, f_p_n, &x, &iterate)
}

/// [`monolithic_evaluate`] at an iterate already in hand, with the stress and the
/// continuum tangent taken from the linearization rather than evaluated again.
fn evaluate<C: ElasticPlastic>(
    model: &C,
    f: &DeformationGradient,
    f_p_n: &DeformationGradientPlastic,
    x: &Unknowns,
    iterate: &Iterate,
) -> Result<Monolithic, ConstitutiveError> {
    let sensitivities = Sensitivities::new(model, f, f_p_n, x, iterate)?;
    let reference = reference(model);
    let (a, b) = (x[SIZE - 1], -iterate.residual[SIZE - 1] / reference);
    let (k_vv, factor) = local_jacobian(&sensitivities.jacobian()?, a, b, reference);
    let mut k_vu = Matrix::zero(SIZE, 9);
    for k in 0..3 {
        for l in 0..3 {
            let d_m = sensitivities
                .linearization
                .mandel_derivative(&basis(k, l), &DeformationGradientPlastic::zero());
            let (d_direction, d_magnitude) = sensitivities.direction_slope(&d_m)?;
            for i in 0..3 {
                for j in 0..3 {
                    k_vu[3 * i + j][3 * k + l] = -a * d_direction[i][j].value();
                }
            }
            k_vu[SIZE - 1][3 * k + l] = factor * d_magnitude;
        }
    }
    let mut k_uv = Matrix::zero(9, SIZE);
    for c in 0..3 {
        for d in 0..3 {
            let d_p = sensitivities
                .linearization
                .stress_derivative(&DeformationGradient::zero(), &sensitivities.slope(c, d));
            for i in 0..3 {
                for j in 0..3 {
                    k_uv[3 * i + j][3 * c + d] = d_p[i][j].value();
                }
            }
        }
    }
    let Linearization {
        stress, tangent, ..
    } = sensitivities.linearization;
    Ok(Monolithic {
        stress,
        residual_local: monolithic_local_residual(iterate, a, reference),
        tangent_uu: tangent,
        tangent_vu: k_vu,
        tangent_uv: k_uv,
        tangent_vv: k_vv,
    })
}
