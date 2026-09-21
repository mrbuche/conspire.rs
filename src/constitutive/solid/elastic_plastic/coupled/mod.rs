#[cfg(test)]
mod test;

use super::{ElasticPlastic, Entries4, Matrix3, entries_4, fischer_burmeister, matrix_3, rank_4};
use crate::{
    constitutive::{ConstitutiveError, fluid::plastic::PlasticStateVariables},
    math::{Matrix, Quantity, Rank2, SquareMatrix, Tensor, Vector},
    mechanics::{
        DeformationGradient, DeformationGradientPlastic, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, FlowDirectionPlastic, MandelStressElastic, Scalar,
    },
};
use std::{array::from_fn, fmt::Debug};

pub(crate) const SIZE: usize = 10;
const MAX_ITERATIONS: usize = 30;
const TOLERANCE: Scalar = 1e-12;
const LOCAL_TOLERANCE: Scalar = 1e-11;
const INITIAL_MULTIPLIER: Scalar = 1e-3;
const ZERO: Matrix3 = [[0.0; 3]; 3];
const EYE: Matrix3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

type Unknowns = [Scalar; SIZE];

fn mul(a: &Matrix3, b: &Matrix3) -> Matrix3 {
    from_fn(|i| from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn transpose(a: &Matrix3) -> Matrix3 {
    from_fn(|i| from_fn(|j| a[j][i]))
}

fn add(a: &Matrix3, b: &Matrix3, scale: Scalar) -> Matrix3 {
    from_fn(|i| from_fn(|j| a[i][j] + scale * b[i][j]))
}

fn symmetric(a: &Matrix3) -> Matrix3 {
    from_fn(|i| from_fn(|j| 0.5 * (a[i][j] + a[j][i])))
}

fn trace(a: &Matrix3) -> Scalar {
    a[0][0] + a[1][1] + a[2][2]
}

fn determinant(a: &Matrix3) -> Scalar {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

fn contract(c: &Entries4, x: &Matrix3) -> Matrix3 {
    from_fn(|i| {
        from_fn(|j| {
            (0..3)
                .map(|k| (0..3).map(|l| c[i][j][k][l] * x[k][l]).sum::<Scalar>())
                .sum()
        })
    })
}

fn basis(a: usize, b: usize) -> Matrix3 {
    from_fn(|i| from_fn(|j| if i == a && j == b { 1.0 } else { 0.0 }))
}

fn failure<C: ElasticPlastic>(model: &C, error: &dyn Debug) -> ConstitutiveError {
    ConstitutiveError::custom(format!("{error:?}"), model)
}

fn increment(x: &Unknowns) -> FlowDirectionPlastic {
    FlowDirectionPlastic::from(from_fn::<_, 3, _>(|i| from_fn::<_, 3, _>(|j| x[3 * i + j])))
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
    tangent: Entries4,
    stress: Matrix3,
    f_p: Matrix3,
    f_p_inverse: Matrix3,
    f_e: Matrix3,
    mandel: Matrix3,
}

impl Linearization {
    fn new<C: ElasticPlastic>(
        model: &C,
        f: &DeformationGradient,
        f_p: &DeformationGradientPlastic,
    ) -> Result<Self, ConstitutiveError> {
        let f_p_inverse = f_p.inverse();
        let stress = matrix_3(&model.first_piola_kirchhoff_stress(f, f_p)?);
        let f_e = matrix_3(&(f * &f_p_inverse));
        let f_p_matrix = matrix_3(f_p);
        let mandel = mul(&mul(&transpose(&f_e), &stress), &transpose(&f_p_matrix));
        Ok(Self {
            tangent: entries_4(&model.first_piola_kirchhoff_tangent_stiffness(f, f_p)?),
            stress,
            f_p: f_p_matrix,
            f_p_inverse: matrix_3(&f_p_inverse),
            f_e,
            mandel,
        })
    }

    fn stress_derivative(&self, d_f: &Matrix3, d_f_p: &Matrix3) -> Matrix3 {
        let by_f = contract(&self.tangent, d_f);
        let by_f_p = contract(&self.tangent, &mul(&self.f_e, d_f_p));
        let correction = mul(
            &mul(&self.stress, &transpose(d_f_p)),
            &transpose(&self.f_p_inverse),
        );
        from_fn(|i| from_fn(|j| by_f[i][j] - by_f_p[i][j] - correction[i][j]))
    }

    fn mandel_derivative(&self, d_f: &Matrix3, d_f_p: &Matrix3) -> Matrix3 {
        let d_f_e = add(
            &mul(d_f, &self.f_p_inverse),
            &mul(&mul(&self.f_e, d_f_p), &self.f_p_inverse),
            -1.0,
        );
        let d_p = self.stress_derivative(d_f, d_f_p);
        let (f_e_t, f_p_t) = (transpose(&self.f_e), transpose(&self.f_p));
        let d_m = add(
            &add(
                &mul(&mul(&transpose(&d_f_e), &self.stress), &f_p_t),
                &mul(&mul(&f_e_t, &d_p), &f_p_t),
                1.0,
            ),
            &mul(&mul(&f_e_t, &self.stress), &transpose(d_f_p)),
            1.0,
        );
        let d_m = add(&d_m, &self.mandel, trace(&mul(&self.f_p_inverse, d_f_p)));
        let scale = determinant(&self.f_p);
        from_fn(|i| from_fn(|j| scale * d_m[i][j]))
    }
}

/// Everything evaluated once per iterate: the plastic deformation gradient, the
/// deviatoric Mandel stress with its unit direction, the residual, and the multiplier
/// with the hardening modulus at its plastic strain, which the Jacobian needs.
struct Iterate {
    plastic: DeformationGradientPlastic,
    unit: Matrix3,
    direction: Matrix3,
    magnitude: Scalar,
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
        let magnitude = deviatoric.norm().value();
        let raw = matrix_3(&deviatoric);
        let unit: Matrix3 = if magnitude > 0.0 {
            from_fn(|i| from_fn(|j| raw[i][j] / magnitude))
        } else {
            ZERO
        };
        let direction = symmetric(&unit);
        let mut residual = [0.0; SIZE];
        (0..3).for_each(|i| {
            (0..3).for_each(|j| residual[3 * i + j] = x[3 * i + j] - x[9] * direction[i][j])
        });
        residual[9] = model
            .yield_function(&deviatoric, Quantity::new(strain_n + x[9]))?
            .value();
        Ok(Self {
            plastic,
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
struct Sensitivities<'a> {
    linearization: Linearization,
    iterate: &'a Iterate,
    slopes: [[Matrix3; 3]; 3],
}

impl<'a> Sensitivities<'a> {
    fn new<C: ElasticPlastic>(
        model: &C,
        f: &DeformationGradient,
        f_p_n: &DeformationGradientPlastic,
        x: &Unknowns,
        iterate: &'a Iterate,
    ) -> Result<Self, ConstitutiveError> {
        let exponential_slope = entries_4(
            &increment(x)
                .dexpm()
                .map_err(|error| failure(model, &error))?,
        );
        let f_p_n = matrix_3(f_p_n);
        Ok(Self {
            linearization: Linearization::new(model, f, &iterate.plastic)?,
            iterate,
            slopes: from_fn(|a| {
                from_fn(|b| {
                    from_fn(|i| {
                        from_fn(|j| {
                            (0..3)
                                .map(|k| exponential_slope[i][k][a][b] * f_p_n[k][j])
                                .sum()
                        })
                    })
                })
            }),
        })
    }

    /// The slope of the symmetrized flow direction and of $`|\mathbf{M}'|`$ along a
    /// Mandel stress increment.
    fn direction_slope(&self, d_m: &Matrix3) -> (Matrix3, Scalar) {
        let Iterate {
            unit, magnitude, ..
        } = self.iterate;
        // the norm is not differentiable where the deviator vanishes, and the flow
        // direction is zero there: the trial state is elastic
        if *magnitude == 0.0 {
            return (ZERO, 0.0);
        }
        let deviatoric = add(d_m, &EYE, -trace(d_m) / 3.0);
        let d_magnitude = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| unit[i][j] * deviatoric[i][j])
                    .sum::<Scalar>()
            })
            .sum();
        let d_unit: Matrix3 =
            from_fn(|i| from_fn(|j| (deviatoric[i][j] - unit[i][j] * d_magnitude) / magnitude));
        (symmetric(&d_unit), d_magnitude)
    }

    /// The Jacobian of the residual with respect to $`(\mathbf{E},\Delta\gamma)`$.
    fn jacobian(&self) -> [[Scalar; SIZE]; SIZE] {
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
                    .mandel_derivative(&ZERO, &self.slopes[a][b]);
                let (d_direction, d_magnitude) = self.direction_slope(&d_m);
                for i in 0..3 {
                    for j in 0..3 {
                        let identity = if i == a && j == b { 1.0 } else { 0.0 };
                        jacobian[3 * i + j][3 * a + b] = identity - gamma * d_direction[i][j];
                    }
                }
                jacobian[9][3 * a + b] = d_magnitude;
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                jacobian[3 * i + j][9] = -self.iterate.direction[i][j];
            }
        }
        jacobian[9][9] = -*hardening_modulus;
        jacobian
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
/// so the flow direction is the end-of-step one. Returns `None` for an elastic step.
pub(super) fn solve<C: ElasticPlastic>(
    model: &C,
    f: &DeformationGradient,
    state: &PlasticStateVariables,
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
    let scale = model.initial_yield_stress().value().max(1.0);
    let size = |r: &Unknowns| {
        r[..9]
            .iter()
            .fold(r[9].abs() / scale, |norm, value| norm.max(value.abs()))
    };
    let direction = {
        let direction = model.flow_direction(&deviatoric)?;
        (&direction + direction.transpose()) * 0.5
    };
    let mut x = [0.0; SIZE];
    (0..3).for_each(|i| {
        (0..3).for_each(|j| x[3 * i + j] = INITIAL_MULTIPLIER * direction[i][j].value())
    });
    x[9] = INITIAL_MULTIPLIER;
    let mut iterate = Iterate::new(model, f, f_p_n, strain_n, &x)?;
    for _ in 0..MAX_ITERATIONS {
        if size(&iterate.residual) < TOLERANCE {
            return Ok(Some(Converged { x, iterate }));
        }
        let matrix = Sensitivities::new(model, f, f_p_n, &x, &iterate)?.jacobian();
        let mut rhs = Vector::zero(SIZE);
        (0..SIZE).for_each(|row| rhs[row] = -iterate.residual[row]);
        let step = SquareMatrix::from(matrix)
            .solve_lu(&rhs)
            .map_err(|error| failure(model, &error))?;
        (0..SIZE).for_each(|row| x[row] += step[row]);
        iterate = Iterate::new(model, f, f_p_n, strain_n, &x)?;
    }
    if size(&iterate.residual) < TOLERANCE {
        return Ok(Some(Converged { x, iterate }));
    }
    Err(ConstitutiveError::custom(
        format!(
            "The coupled return mapping did not converge, |R| = {:.3e}.",
            size(&iterate.residual)
        ),
        model,
    ))
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
        model.initial_yield_stress().value(),
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
#[allow(clippy::type_complexity)]
pub(super) fn monolithic_tangents<C: ElasticPlastic>(
    model: &C,
    f: &DeformationGradient,
    state: &PlasticStateVariables,
    local: &Vector,
) -> Result<(FirstPiolaKirchhoffTangentStiffness, Matrix, Matrix, Matrix), ConstitutiveError> {
    let (_, _, k_uu, k_vu, k_uv, k_vv) = monolithic_evaluate(model, f, state, local)?;
    Ok((k_uu, k_vu, k_uv, k_vv))
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

/// The local residual and the local block $`K_{vv}`$ alone, all a Newton iteration on the
/// local unknowns needs.
fn monolithic_local<C: ElasticPlastic>(
    model: &C,
    f: &DeformationGradient,
    state: &PlasticStateVariables,
    x: &Unknowns,
) -> Result<(Vector, Matrix), ConstitutiveError> {
    let (f_p_n, &strain_n): (&DeformationGradientPlastic, &Quantity) = state.into();
    let iterate = Iterate::new(model, f, f_p_n, strain_n.value(), x)?;
    let reference = model.initial_yield_stress().value();
    let residual = monolithic_local_residual(&iterate, x[SIZE - 1], reference);
    if residual.iter().all(|entry| entry.abs() < LOCAL_TOLERANCE) {
        return Ok((residual, Matrix::zero(0, 0)));
    }
    let sensitivities = Sensitivities::new(model, f, f_p_n, x, &iterate)?;
    let b = -iterate.residual[SIZE - 1] / reference;
    let (k_vv, _) = local_jacobian(&sensitivities.jacobian(), x[SIZE - 1], b, reference);
    Ok((residual, k_vv))
}

/// The stress, the consistent tangent and the updated plastic state of one step, from a
/// local solve of the monolithic system's unknowns $`(\mathbf{E},\Delta\gamma)`$ alone
/// followed by the Schur complement that eliminates them,
/// $`\mathcal{C}_\mathrm{eff} = K_{uu} - K_{uv}K_{vv}^{-1}K_{vu}`$.
///
/// This is the condensed strategy of the block solver at one integration point, with
/// no state carried between calls. An elastic step has nothing to solve for.
#[allow(clippy::type_complexity)]
pub(crate) fn condensed<C: ElasticPlastic>(
    model: &C,
    f: &DeformationGradient,
    state: &PlasticStateVariables,
) -> Result<
    (
        FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness,
        PlasticStateVariables,
    ),
    ConstitutiveError,
> {
    let (f_p_n, &strain_n): (&DeformationGradientPlastic, &Quantity) = state.into();
    let deviatoric = model.mandel_stress(f, f_p_n)?.deviatoric();
    if model
        .yield_function(&deviatoric, Quantity::new(strain_n.value()))?
        .value()
        <= 0.0
    {
        return Ok((
            model.first_piola_kirchhoff_stress(f, f_p_n)?,
            model.first_piola_kirchhoff_tangent_stiffness(f, f_p_n)?,
            state.clone(),
        ));
    }
    let direction = {
        let direction = model.flow_direction(&deviatoric)?;
        (&direction + direction.transpose()) * 0.5
    };
    let mut x = [0.0; SIZE];
    (0..3).for_each(|i| {
        (0..3).for_each(|j| x[3 * i + j] = INITIAL_MULTIPLIER * direction[i][j].value())
    });
    x[SIZE - 1] = INITIAL_MULTIPLIER;
    let mut local = Vector::from(x.to_vec());
    let mut converged = false;
    for _ in 0..MAX_ITERATIONS {
        let unknowns: Unknowns = from_fn(|i| local[i]);
        let (residual, k_vv) = monolithic_local(model, f, state, &unknowns)?;
        if residual.iter().all(|entry| entry.abs() < LOCAL_TOLERANCE) {
            converged = true;
            break;
        }
        let step = k_vv
            .iter()
            .cloned()
            .collect::<SquareMatrix>()
            .solve_lu(&residual)
            .map_err(|error| failure(model, &error))?;
        (0..SIZE).for_each(|i| local[i] -= step[i])
    }
    if !converged {
        return Err(failure(model, &"the local solve did not converge"));
    }
    let (stress, _, tangent, k_vu, k_uv, k_vv) = monolithic_evaluate(model, f, state, &local)?;
    let lu = k_vv
        .iter()
        .cloned()
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
    let continuum = entries_4(&tangent);
    let entries: Entries4 = from_fn(|i| {
        from_fn(|j| {
            from_fn(|k| {
                from_fn(|l| {
                    continuum[i][j][k][l]
                        - (0..SIZE)
                            .map(|m| k_uv[3 * i + j][m] * solved[3 * k + l][m])
                            .sum::<Scalar>()
                })
            })
        })
    });
    Ok((
        stress,
        rank_4(&entries),
        monolithic_state(model, state, &local)?,
    ))
}

/// Everything the monolithic system needs at a point from one evaluation: the first
/// Piola-Kirchhoff stress at the trial plastic state, the local residual, and the
/// tangent blocks $`(K_{uu},K_{vu},K_{uv},K_{vv})`$ of [`monolithic_tangents`].
#[allow(clippy::type_complexity)]
pub(crate) fn monolithic_evaluate<C: ElasticPlastic>(
    model: &C,
    f: &DeformationGradient,
    state: &PlasticStateVariables,
    local: &Vector,
) -> Result<
    (
        FirstPiolaKirchhoffStress,
        Vector,
        FirstPiolaKirchhoffTangentStiffness,
        Matrix,
        Matrix,
        Matrix,
    ),
    ConstitutiveError,
> {
    let (f_p_n, &strain_n): (&DeformationGradientPlastic, &Quantity) = state.into();
    let x: Unknowns = from_fn(|i| local[i]);
    let iterate = Iterate::new(model, f, f_p_n, strain_n.value(), &x)?;
    let sensitivities = Sensitivities::new(model, f, f_p_n, &x, &iterate)?;
    let reference = model.initial_yield_stress().value();
    let (a, b) = (x[SIZE - 1], -iterate.residual[SIZE - 1] / reference);
    let (k_vv, factor) = local_jacobian(&sensitivities.jacobian(), a, b, reference);
    let mut k_vu = Matrix::zero(SIZE, 9);
    for k in 0..3 {
        for l in 0..3 {
            let d_m = sensitivities
                .linearization
                .mandel_derivative(&basis(k, l), &ZERO);
            let (d_direction, d_magnitude) = sensitivities.direction_slope(&d_m);
            for i in 0..3 {
                for j in 0..3 {
                    k_vu[3 * i + j][3 * k + l] = -a * d_direction[i][j];
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
                .stress_derivative(&ZERO, &sensitivities.slopes[c][d]);
            for i in 0..3 {
                for j in 0..3 {
                    k_uv[3 * i + j][3 * c + d] = d_p[i][j];
                }
            }
        }
    }
    Ok((
        model.first_piola_kirchhoff_stress(f, &iterate.plastic)?,
        monolithic_local_residual(&iterate, x[SIZE - 1], reference),
        model.first_piola_kirchhoff_tangent_stiffness(f, &iterate.plastic)?,
        k_vu,
        k_uv,
        k_vv,
    ))
}
