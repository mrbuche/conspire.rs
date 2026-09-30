use super::{
    ElasticPlastic, SIZE, Unknowns, basis, failure, increment,
    linearization::Linearization,
    reference,
    sensitivities::{Iterate, Sensitivities},
};
use crate::{
    constitutive::{ConstitutiveError, fluid::plastic::PlasticStateVariables},
    math::{Matrix, Quantity, TensorArray, Vector},
    mechanics::{
        DeformationGradient, DeformationGradientPlastic, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, Scalar,
    },
};
use std::array::from_fn;

/// The plastic deformation gradient of a monolithic trial state: the local unknowns
/// are $`(\mathbf{E},\Delta\gamma)`$ and $`\mathbf{F}_\mathrm{p}=\exp(\mathbf{E})\mathbf{F}_\mathrm{p}^n`$.
pub(crate) fn monolithic_plastic<C: ElasticPlastic>(
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

/// The local residual of the monolithic system: the flow rule as in
/// [`solve`](super::local::solve), and the yield inequality imposed as the
/// Fischer-Burmeister complementarity $`\varphi(\Delta\gamma,-f/Y_0)`$, so an elastic
/// step recovers $`\Delta\gamma=0`$ on its own.
pub(super) fn monolithic_local_residual(
    iterate: &Iterate,
    gamma: Scalar,
    reference: Scalar,
) -> Vector {
    let mut residual = Vector::zero(SIZE);
    (0..SIZE - 1).for_each(|row| residual[row] = iterate.residual[row]);
    residual[SIZE - 1] =
        super::super::fischer_burmeister(gamma, -iterate.residual[SIZE - 1] / reference);
    residual
}

pub(crate) fn monolithic_residual_local<C: ElasticPlastic>(
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
/// The local residual is the coupled one of [`solve`](super::local::solve) with its yield
/// row replaced by the Fischer-Burmeister function, so those blocks are the coupled
/// Jacobian's with that row scaled by $`\partial\varphi/\partial b\,(-1/Y_0)`$ and the
/// multiplier column carrying the extra $`\partial\varphi/\partial a`$. $`K_{uu}`$ is the
/// continuum tangent, since the flow direction depends on $`\mathbf{F}`$ only through the
/// local unknowns.
pub(crate) fn monolithic_tangents<C: ElasticPlastic>(
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
pub(super) fn local_jacobian(
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
pub(super) fn evaluate<C: ElasticPlastic>(
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
