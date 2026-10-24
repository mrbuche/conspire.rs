use super::super::{
    super::{
        HessianBlock, Jacobian, LuDecomposition, Matrix, Scalar, Solution, SquareMatrix, Tensor,
        Vector,
        sparse::{CscMatrix, SparseSolver},
    },
    Direct, OptimizationError, SolveStrategy, TrustRegion,
};
use std::{fmt::Display, ops::Mul};

use super::NewtonRaphson;
use super::adaptive::adaptive_step_blocked;
use super::backtrack::{Backtrack, half_square, residual_step};
use super::convergence::{converged, limit_decrement};
use super::kkt::{converge_local, eliminate_blocks, kkt_block, kkt_entry, kkt_residual};
use super::penalty::{backtrack_penalty, merit_slope, raise_penalty, violation};

#[expect(clippy::too_many_arguments)]
pub(super) fn blocked<B, S, U, V, Ru, Rv, Kuu, Kvu, Kuv, Kvv>(
    newton_raphson: &NewtonRaphson<Direct, S>,
    search: B,
    mut function: impl FnMut(&U, &V) -> Result<Scalar, String>,
    mut residual_global: impl FnMut(&U, &V) -> Result<Ru, String>,
    mut residual_local: impl FnMut(&U, &V) -> Result<Rv, String>,
    mut tangents: impl FnMut(&U, &V) -> Result<(Kuu, Kvu, Kuv, Kvv), String>,
    initial_guess: (U, V),
    constraint_global: (CscMatrix, Vector),
    constraint_local: (CscMatrix, Vector),
    sparse: Option<SparseSolver>,
    strategy: SolveStrategy,
) -> Result<(U, V), OptimizationError>
where
    B: Backtrack,
    S: Display,
    U: Solution,
    V: Solution,
    Ru: Jacobian,
    Rv: Jacobian,
    Kuu: HessianBlock,
    Kvu: HessianBlock,
    Kuv: HessianBlock,
    Kvv: HessianBlock,
    for<'a> &'a CscMatrix: Mul<&'a U, Output = Vector> + Mul<&'a V, Output = Vector>,
{
    let (mut global, mut local) = initial_guess;
    let mut penalty = 0.0 as Scalar;
    let (constraint_matrix_global, constraint_rhs_global) = constraint_global;
    let (constraint_matrix_local, constraint_rhs_local) = constraint_local;
    let num_global = global.size();
    let num_local = local.size();
    let num_outer = num_global + constraint_rhs_global.len();
    let num_inner = num_local + constraint_rhs_local.len();
    let mut multipliers_global = Vector::zero(constraint_rhs_global.len());
    let mut multipliers_local = Vector::zero(constraint_rhs_local.len());
    let eliminating = !matches!(strategy, SolveStrategy::Monolithic { elimination: false });
    let condensed = match strategy {
        SolveStrategy::Condensed(ref local_solver) => Some(local_solver),
        SolveStrategy::Monolithic { .. } => None,
    };
    //
    // With a sparse solver the local block is eliminated as the independent blocks it
    // is held as, never as one matrix, so nothing dense in both the global and the
    // local unknowns is ever formed.
    //
    let structured = sparse.is_some() && eliminating && condensed.is_none();
    if sparse.is_some() && eliminating && !structured {
        unimplemented!("Condensing the local block sparsely is not supported.")
    }
    if structured {
        assert!(
            constraint_rhs_local.len() == 0,
            "Eliminating the local block sparsely does not support constraints on it."
        )
    }
    let (inner, outer) = if eliminating && !structured {
        (num_inner, num_outer)
    } else {
        (0, 0)
    };
    let whole = if eliminating || sparse.is_some() {
        0
    } else {
        num_outer + num_inner
    };
    let mut column = Vector::zero(inner);
    let mut coupling_global = Matrix::zero(outer, inner);
    let mut coupling_local = Matrix::zero(inner, outer);
    let mut decrement_inner = Vector::zero(num_inner);
    let mut decrement_outer = Vector::zero(num_outer);
    let mut decrement_whole = Vector::zero(whole);
    let mut eliminated = vec![Vector::zero(inner); outer.min(num_global)];
    let mut factorization = LuDecomposition::zero(inner);
    let mut factorization_outer = LuDecomposition::zero(outer);
    let mut factorization_whole = LuDecomposition::zero(whole);
    let mut monolithic = SquareMatrix::zero(whole);
    let mut residual = Vector::zero(num_outer + num_inner);
    let mut scales_inner = None;
    let mut scales_outer = None;
    let mut tangent_inner = SquareMatrix::zero(inner);
    let mut tangent_outer = SquareMatrix::zero(outer);
    let mut update_inner = Vector::zero(num_inner);
    let mut update_outer = Vector::zero(num_outer);
    let mut schur = sparse
        .as_ref()
        .filter(|_| structured)
        .map(|solver| CscMatrix::from_pattern(num_outer, num_outer, solver.pattern().to_vec()));
    let (mut radius, max_radius) =
        if let TrustRegion::Adaptive { radius, max_radius } = newton_raphson.trust_region {
            (radius, max_radius)
        } else {
            (0.0, 0.0)
        };
    let adaptive = B::OBJECTIVE
        && matches!(newton_raphson.trust_region, TrustRegion::Adaptive { .. })
        && eliminating
        && !structured
        && condensed.is_none();
    let mut steps = 0;
    loop {
        if let Some(local_solver) = condensed {
            converge_local(
                local_solver,
                &mut residual_local,
                &mut tangents,
                &global,
                &mut local,
                &mut multipliers_local,
                &constraint_matrix_local,
                &constraint_rhs_local,
                num_local,
                &mut update_inner,
                &mut tangent_inner,
                &mut factorization,
            )?
        }
        kkt_residual(
            residual_global(&global, &local)?,
            &multipliers_global,
            &constraint_matrix_global,
            &constraint_rhs_global,
            &global,
            &mut update_outer,
        );
        kkt_residual(
            residual_local(&global, &local)?,
            &multipliers_local,
            &constraint_matrix_local,
            &constraint_rhs_local,
            &local,
            &mut update_inner,
        );
        update_outer
            .iter()
            .chain(update_inner.iter())
            .zip(residual.iter_mut())
            .for_each(|(entry, residual_i)| *residual_i = *entry);
        let converged_outer =
            converged(newton_raphson, &update_outer, num_global, &mut scales_outer);
        let converged_inner =
            converged(newton_raphson, &update_inner, num_local, &mut scales_inner);
        if converged_outer && converged_inner {
            return Ok((global, local));
        } else if steps == newton_raphson.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                newton_raphson.max_steps,
                format!("{newton_raphson:?}"),
            ));
        }
        steps += 1;
        let (tangent_uu, tangent_vu, tangent_uv, tangent_vv) = tangents(&global, &local)?;
        if structured {
            eliminate_blocks(
                sparse.as_ref().unwrap(),
                schur.as_mut().unwrap(),
                &tangent_uu,
                &tangent_vu,
                &tangent_uv,
                &tangent_vv,
                &constraint_matrix_global,
                (num_global, num_outer, num_local),
                &mut update_outer,
                &update_inner,
                &mut decrement_outer,
                &mut decrement_inner,
            )?;
        } else if eliminating {
            kkt_block(
                &tangent_uu,
                &constraint_matrix_global,
                num_global,
                &mut tangent_outer,
                0,
            );
            kkt_block(
                &tangent_vv,
                &constraint_matrix_local,
                num_local,
                &mut tangent_inner,
                0,
            );
            tangent_uv.fill_into_block(&mut coupling_global, 0, 0);
            tangent_vu.fill_into_block(&mut coupling_local, 0, 0);
            tangent_inner.factorize_lu_into(&mut factorization)?;
            eliminated
                .iter_mut()
                .enumerate()
                .for_each(|(k, eliminated_k)| {
                    (0..num_local).for_each(|i| column[i] = coupling_local[i][k]);
                    factorization.solve_into(&column, eliminated_k)
                });
            factorization.solve_into(&update_inner, &mut decrement_inner);
            (0..num_global).for_each(|i| {
                (0..num_local).for_each(|j| {
                    let coupling = coupling_global[i][j];
                    (0..num_global)
                        .for_each(|k| tangent_outer[i][k] -= coupling * eliminated[k][j]);
                    update_outer[i] -= coupling * decrement_inner[j]
                })
            });
            if adaptive {
                adaptive_step_blocked(
                    newton_raphson,
                    &mut function,
                    &mut residual_global,
                    &mut residual_local,
                    &mut global,
                    &mut local,
                    &mut multipliers_global,
                    &mut multipliers_local,
                    &mut decrement_outer,
                    &mut decrement_inner,
                    &mut penalty,
                    &mut radius,
                    max_radius,
                    &tangent_outer,
                    &update_outer,
                    &coupling_global,
                    &eliminated,
                    (&constraint_matrix_global, &constraint_rhs_global),
                    (&constraint_matrix_local, &constraint_rhs_local),
                )?;
                continue;
            }
            tangent_outer.factorize_lu_into(&mut factorization_outer)?;
            factorization_outer.solve_into(&update_outer, &mut decrement_outer);
            (0..num_global).for_each(|k| {
                decrement_inner
                    .iter_mut()
                    .zip(eliminated[k].iter())
                    .for_each(|(decrement_inner_i, eliminated_ki)| {
                        *decrement_inner_i -= eliminated_ki * decrement_outer[k]
                    })
            });
        } else {
            if sparse.is_none() {
                kkt_block(
                    &tangent_uu,
                    &constraint_matrix_global,
                    num_global,
                    &mut monolithic,
                    0,
                );
                kkt_block(
                    &tangent_vv,
                    &constraint_matrix_local,
                    num_local,
                    &mut monolithic,
                    num_outer,
                );
            }
            if let Some(ref solver) = sparse {
                //
                // The block layout is the same either way, so the entry a
                // sparse solver asks for is read from whichever block holds it.
                //
                decrement_whole = solver.solve(
                    |i, j| {
                        kkt_entry(
                            i,
                            j,
                            num_global,
                            num_outer,
                            num_local,
                            &tangent_uu,
                            &tangent_vu,
                            &tangent_uv,
                            &tangent_vv,
                            &constraint_matrix_global,
                            &constraint_matrix_local,
                        )
                    },
                    &residual,
                )?
            } else {
                tangent_uv.fill_into_block(&mut monolithic, 0, num_outer);
                tangent_vu.fill_into_block(&mut monolithic, num_outer, 0);
                monolithic.factorize_lu_into(&mut factorization_whole)?;
                factorization_whole.solve_into(&residual, &mut decrement_whole)
            }
            decrement_whole
                .iter()
                .zip(decrement_outer.iter_mut().chain(decrement_inner.iter_mut()))
                .for_each(|(entry, decrement_i)| *decrement_i = *entry);
        }
        let fraction = limit_decrement(
            newton_raphson,
            &mut [
                (&mut decrement_outer, num_global),
                (&mut decrement_inner, num_local),
            ],
        );
        let step_size = if !B::SEARCHES {
            1.0
        } else if !B::OBJECTIVE {
            //
            // A root has no objective, so the trial point is judged by the
            // residual it leaves. Condensed makes the local residual vanish at
            // every trial, so only the global residual is measured then.
            //

            residual_step(
                newton_raphson,
                &search,
                || {
                    half_square(
                        residual
                            .iter()
                            .take(if condensed.is_some() {
                                num_outer
                            } else {
                                residual.len()
                            })
                            .copied(),
                    )
                },
                |trial_size| {
                    let mut trial_global = global.clone();
                    let mut trial_local = local.clone();
                    let mut trial_multipliers_global = multipliers_global.clone();
                    let mut trial_multipliers_local = multipliers_local.clone();
                    trial_global.decrement_from_chained(
                        &mut trial_multipliers_global,
                        &(&decrement_outer * trial_size),
                    );
                    if let Some(local_solver) = condensed {
                        converge_local(
                            local_solver,
                            &mut residual_local,
                            &mut tangents,
                            &trial_global,
                            &mut trial_local,
                            &mut trial_multipliers_local,
                            &constraint_matrix_local,
                            &constraint_rhs_local,
                            num_local,
                            &mut update_inner,
                            &mut tangent_inner,
                            &mut factorization,
                        )
                        .map_err(|error| format!("{error}"))?
                    } else {
                        trial_local.decrement_from_chained(
                            &mut trial_multipliers_local,
                            &(&decrement_inner * trial_size),
                        )
                    }
                    let mut trial_outer = Vector::zero(num_outer);
                    kkt_residual(
                        residual_global(&trial_global, &trial_local)?,
                        &trial_multipliers_global,
                        &constraint_matrix_global,
                        &constraint_rhs_global,
                        &trial_global,
                        &mut trial_outer,
                    );
                    let mut merit = half_square(trial_outer.iter().copied());
                    if condensed.is_none() {
                        let mut trial_inner = Vector::zero(num_inner);
                        kkt_residual(
                            residual_local(&trial_global, &trial_local)?,
                            &trial_multipliers_local,
                            &constraint_matrix_local,
                            &constraint_rhs_local,
                            &trial_local,
                            &mut trial_inner,
                        );
                        merit += half_square(trial_inner.iter().copied())
                    }
                    if !B::MERIT {
                        tangents(&trial_global, &trial_local)?;
                    }
                    Ok(merit)
                },
                fraction,
            )?
        } else {
            penalty = raise_penalty(
                penalty,
                multipliers_global
                    .iter()
                    .zip(decrement_outer.iter().skip(num_global))
                    .chain(
                        multipliers_local
                            .iter()
                            .zip(decrement_inner.iter().skip(num_local)),
                    ),
            );
            let violated = penalty
                * (violation(&constraint_matrix_global, &constraint_rhs_global, &global)
                    + violation(&constraint_matrix_local, &constraint_rhs_local, &local));
            let mut gradient_global = Vector::zero(num_global);
            let mut gradient_local = Vector::zero(num_local);
            residual_global(&global, &local)?.fill_into(&mut gradient_global);
            residual_local(&global, &local)?.fill_into(&mut gradient_local);
            let slope = merit_slope(
                gradient_global
                    .iter()
                    .zip(decrement_outer.iter())
                    .chain(gradient_local.iter().zip(decrement_inner.iter())),
                violated,
            );
            let value = function(&global, &local)? + violated;
            backtrack_penalty(
                newton_raphson,
                &search,
                |step| {
                    let mut trial_global = global.clone();
                    let mut trial_local = local.clone();
                    let mut trial_multipliers_global = multipliers_global.clone();
                    let mut trial_multipliers_local = multipliers_local.clone();
                    trial_global.decrement_from_chained(
                        &mut trial_multipliers_global,
                        &(&decrement_outer * step),
                    );
                    //
                    // Condensed makes the local variables a function of the
                    // global ones, so a trial point is where they solve to,
                    // not where the increment predicted they would.
                    //
                    if let Some(local_solver) = condensed {
                        converge_local(
                            local_solver,
                            &mut residual_local,
                            &mut tangents,
                            &trial_global,
                            &mut trial_local,
                            &mut trial_multipliers_local,
                            &constraint_matrix_local,
                            &constraint_rhs_local,
                            num_local,
                            &mut update_inner,
                            &mut tangent_inner,
                            &mut factorization,
                        )
                        .map_err(|error| format!("{error}"))?
                    } else {
                        trial_local.decrement_from_chained(
                            &mut trial_multipliers_local,
                            &(&decrement_inner * step),
                        )
                    }
                    Ok(function(&trial_global, &trial_local)?
                        + penalty
                            * (violation(
                                &constraint_matrix_global,
                                &constraint_rhs_global,
                                &trial_global,
                            ) + violation(
                                &constraint_matrix_local,
                                &constraint_rhs_local,
                                &trial_local,
                            )))
                },
                value,
                slope,
            )?
        };
        if step_size == 1.0 {
            global.decrement_from_chained(&mut multipliers_global, &decrement_outer);
            local.decrement_from_chained(&mut multipliers_local, &decrement_inner)
        } else {
            global.decrement_from_chained(&mut multipliers_global, &(&decrement_outer * step_size));
            local.decrement_from_chained(&mut multipliers_local, &(&decrement_inner * step_size))
        }
    }
}
