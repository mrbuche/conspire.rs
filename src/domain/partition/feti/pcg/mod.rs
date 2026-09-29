use std::cell::Cell;

use crate::{
    domain::feti::{
        THREADS,
        dual::{dual_action, dual_operator, dual_precondition, dual_precondition_dirichlet},
        dual_primal::{coarse::Coarse, rigid_projector::RigidProjector},
        subdomain::Subdomain,
    },
    math::{
        Scalar, Tensor, Vector,
        optimize::{Krylov, KrylovError, KrylovMethod},
    },
};

/// Solves the dual (interface) problem `(F + C S_pp^-1 C^T) . lambda = rhs`
/// by conjugate gradients, Dirichlet-preconditioned. For a symmetric positive
/// definite tangent the operator is SPD given the corners are pinned, so this
/// needs no projection against a rigid-body null space the way plain FETI
/// would; a nonsymmetric tangent needs GMRES instead. Dirichlet is the default over lumped: its
/// condition-number bound is near mesh-independent (`1 + log(H/h)^2`) where
/// lumped's degrades with the subdomain-to-mesh-size ratio, at the price of
/// one local interior solve per subdomain per iteration, so it wins as soon
/// as a subdomain has more than a handful of elements per edge, the
/// realistic regime. `dual_precondition` (lumped) stays available for the
/// pathologically-small-subdomain case.
pub(crate) fn projected_pcg<B>(
    subdomains: &[Subdomain<B>],
    coarse: &Coarse,
    rhs: &Vector,
) -> Result<Vector, KrylovError>
where
    B: Sync,
{
    projected_pcg_with(
        subdomains,
        coarse,
        rhs,
        Preconditioner::Dirichlet,
        Krylov::default().rel_tol,
        KrylovMethod::default(),
    )
}

/// Which FETI-DP preconditioner the dual PCG applies.
///
/// `Dirichlet` is the default: its condition-number bound is near
/// mesh-independent, at the price of one interior solve per subdomain per
/// iteration over `Lumped`'s cheap matvec — see `projected_pcg`'s own doc
/// for the tradeoff in full.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preconditioner {
    /// `sum_s B_s K_dd,s B_s^T` — a local matvec in place of a local solve.
    Lumped,
    /// `sum_s B_b,s S_s B_b,s^T` — near mesh-independent convergence.
    Dirichlet,
}

pub(crate) fn projected_pcg_with<B>(
    subdomains: &[Subdomain<B>],
    coarse: &Coarse,
    rhs: &Vector,
    preconditioner: Preconditioner,
    rel_tol: Scalar,
    method: KrylovMethod,
) -> Result<Vector, KrylovError>
where
    B: Sync,
{
    projected_pcg_counting(
        subdomains,
        coarse,
        rhs,
        preconditioner,
        rel_tol,
        method,
        &Cell::new(0),
    )
}

/// As `projected_pcg_with`, adding each application of the dual operator to
/// `applications`.
pub(crate) fn projected_pcg_counting<B>(
    subdomains: &[Subdomain<B>],
    coarse: &Coarse,
    rhs: &Vector,
    preconditioner: Preconditioner,
    rel_tol: Scalar,
    method: KrylovMethod,
    applications: &Cell<usize>,
) -> Result<Vector, KrylovError>
where
    B: Sync,
{
    Krylov {
        rel_tol,
        method,
        ..Krylov::default()
    }
    .solve(
        |lambda| {
            applications.set(applications.get() + 1);
            dual_operator(subdomains, lambda, coarse, THREADS)
        },
        |lambda: &Vector| match preconditioner {
            Preconditioner::Lumped => dual_precondition(subdomains, lambda, THREADS),
            Preconditioner::Dirichlet => dual_precondition_dirichlet(subdomains, lambda, THREADS),
        },
        rhs,
    )
}

/// Solves the dual problem of classical FETI, `F lambda - G alpha = d` with
/// `G^T lambda = e`, for the multipliers and the rigid-body amplitudes.
///
/// The multipliers are `lambda_0 + mu`, with `lambda_0 = G (G^T G)^-1 e`
/// satisfying the constraint and `mu` found by the same Krylov solve as
/// FETI-DP on `P F P mu = P (d - F lambda_0)`, preconditioned by `P M P`. The
/// projector `P` keeps every iterate in the space where `G^T mu = 0`, which is
/// what makes the floating subdomains' local solves consistent.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rigid_projected_pcg<B>(
    subdomains: &[Subdomain<B>],
    projector: &RigidProjector,
    d: &Vector,
    e: &Vector,
    preconditioner: Preconditioner,
    rel_tol: Scalar,
    method: KrylovMethod,
    applications: &Cell<usize>,
) -> Result<(Vector, Vector), KrylovError>
where
    B: Sync,
{
    let particular = projector.particular(e, d.len());
    let rhs = projector.project(&(d.clone() - dual_action(subdomains, &particular, THREADS)));
    let correction = Krylov {
        rel_tol,
        method,
        ..Krylov::default()
    }
    .solve(
        |mu: &Vector| {
            applications.set(applications.get() + 1);
            projector.project(&dual_action(subdomains, &projector.project(mu), THREADS))
        },
        |residual: &Vector| {
            let residual = projector.project(residual);
            projector.project(&match preconditioner {
                Preconditioner::Lumped => dual_precondition(subdomains, &residual, THREADS),
                Preconditioner::Dirichlet => {
                    dual_precondition_dirichlet(subdomains, &residual, THREADS)
                }
            })
        },
        &rhs,
    )?;
    let lambda = particular + projector.project(&correction);
    let alpha = projector.amplitudes(&(dual_action(subdomains, &lambda, THREADS) - d.clone()));
    Ok((lambda, alpha))
}

/// Recovers each subdomain's full local solution (corner and dual DOFs
/// together) once the coarse (corner) solution and multipliers are known:
/// `u_d,s = K_dd,s^-1 (f_d,s - K_dp,s . u_p - B_s^T . lambda)`, with `u_p`
/// gathered into this subdomain's raw local corner positions and the
/// coupling term `K_dd,s^-1 K_dp,s . u_p = dual_map_s . u_p` already
/// available from `condense()`.
pub(crate) fn primal_recovery<B>(
    subdomains: &[Subdomain<B>],
    local_forces: &[Vector],
    corner_solution: &Vector,
    lambda: &Vector,
) -> Vec<Vector> {
    subdomains
        .iter()
        .zip(local_forces.iter())
        .map(|(subdomain, local_force)| {
            let multiplier_rhs = subdomain
                .interface()
                .apply_transpose(lambda, subdomain.num_local());
            let combined_rhs = local_force - &multiplier_rhs;
            let dual_solution = subdomain.local_solve(&combined_rhs);
            let primal_local = subdomain.gather_primal(corner_solution);
            let coupling_correction =
                subdomain.scatter_dual(&(subdomain.dual_map() * &primal_local));
            dual_solution - coupling_correction + subdomain.scatter_primal(&primal_local)
        })
        .collect()
}
