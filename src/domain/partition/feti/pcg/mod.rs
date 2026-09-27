use crate::domain::partition::feti::{
    THREADS,
    dual::{dual_operator, dual_precondition, dual_precondition_dirichlet},
    dual_primal::coarse::Coarse,
    subdomain::Subdomain,
};
use crate::math::{
    Scalar, Tensor, Vector,
    optimize::{Krylov, KrylovError},
};

/// Solves the dual (interface) problem `(F + C S_pp^-1 C^T) . lambda = rhs`
/// by conjugate gradients, Dirichlet-preconditioned — SPD given the corners
/// are pinned, so this needs no projection against a rigid-body null space
/// the way plain FETI would. Dirichlet is the default over lumped: its
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
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preconditioner {
    Lumped,
    Dirichlet,
}

pub(crate) fn projected_pcg_with<B>(
    subdomains: &[Subdomain<B>],
    coarse: &Coarse,
    rhs: &Vector,
    preconditioner: Preconditioner,
    rel_tol: Scalar,
) -> Result<Vector, KrylovError>
where
    B: Sync,
{
    Krylov {
        rel_tol,
        ..Krylov::default()
    }
    .solve(
        |lambda| dual_operator(subdomains, lambda, coarse, THREADS),
        |lambda: &Vector| match preconditioner {
            Preconditioner::Lumped => dual_precondition(subdomains, lambda, THREADS),
            Preconditioner::Dirichlet => dual_precondition_dirichlet(subdomains, lambda, THREADS),
        },
        rhs,
    )
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
            let combined_rhs: Vector = local_force
                .iter()
                .zip(multiplier_rhs.iter())
                .map(|(&f, &m)| f - m)
                .collect();
            let dual_solution = subdomain.local_solve(&combined_rhs);
            let primal_local = subdomain.gather_primal(corner_solution);
            let coupling_correction =
                subdomain.scatter_dual(&(subdomain.dual_map() * &primal_local));
            dual_solution - coupling_correction + subdomain.scatter_primal(&primal_local)
        })
        .collect()
}
