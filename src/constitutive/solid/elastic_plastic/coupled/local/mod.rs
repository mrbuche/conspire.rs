use super::{
    ElasticPlastic, SIZE, failure,
    monolithic::{Monolithic, evaluate, local_jacobian, monolithic_local_residual},
    reference,
    sensitivities::{Iterate, Sensitivities},
};
use crate::{
    constitutive::{ConstitutiveError, fluid::plastic::PlasticStateVariables},
    math::{
        Quantity, Rank2, SquareMatrix, Vector,
        optimize::{NewtonRaphson, converged, limit_decrement},
    },
    mechanics::{
        DeformationGradient, DeformationGradientPlastic, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, Scalar,
    },
};

const INITIAL_MULTIPLIER: Scalar = 1e-3;

/// A converged step: the unknowns and the state they were evaluated at.
pub(crate) struct Converged {
    x: [Scalar; SIZE],
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
/// [`return_map`](super::super::ElasticPlastic::return_map) and [`condensed`]. The
/// monolithic strategies do not call it: they step the deformation gradient and these
/// unknowns together. It is converged and limited as `local_solver` says. Returns
/// `None` for an elastic step.
pub(crate) fn solve<C: ElasticPlastic>(
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

pub(crate) fn updated_state(
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
