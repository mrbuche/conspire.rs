use crate::domain::partition::feti::{
    dual_primal::coarse::Coarse, parallel::thread_count, subdomain::Subdomain,
};
use crate::math::{Tensor, Vector};
use std::thread::scope;

/// Reduces a per-subdomain local step (`local_solve` for the dual operator,
/// `local_apply` for the lumped preconditioner) into a multiplier-space
/// vector: matrix-free, one independent local step per subdomain plus a
/// reduction, never assembled.
pub(crate) fn dual_reduce<B>(
    subdomains: &[Subdomain<B>],
    lambda: &Vector,
    max_threads: usize,
    local_step: impl Fn(&Subdomain<B>, &Vector) -> Vector + Sync,
) -> Vector
where
    B: Sync,
{
    let num_multipliers = lambda.len();
    let reduce = |chunk: &[Subdomain<B>]| {
        chunk
            .iter()
            .map(|subdomain| {
                let rhs = subdomain
                    .interface()
                    .apply_transpose(lambda, subdomain.num_local());
                let local = local_step(subdomain, &rhs);
                subdomain.interface().apply(&local, num_multipliers)
            })
            .fold(Vector::zero(num_multipliers), |sum, contribution| {
                sum + contribution
            })
    };
    let threads = thread_count(max_threads).min(subdomains.len());
    if threads <= 1 {
        return reduce(subdomains);
    }
    let chunk_size = subdomains.len().div_ceil(threads);
    let partials: Vec<Vector> = scope(|scope| {
        subdomains
            .chunks(chunk_size)
            .map(|chunk| scope.spawn(move || reduce(chunk)))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().expect("subdomain reduction thread panicked"))
            .collect()
    });
    partials
        .into_iter()
        .fold(Vector::zero(num_multipliers), |sum, partial| sum + partial)
}

/// The action of the FETI-DP dual operator `F = sum_s B_s K_dd,s^-1 B_s^T` on
/// a multiplier vector.
pub(crate) fn dual_action<B>(subdomains: &[Subdomain<B>], lambda: &Vector, threads: usize) -> Vector
where
    B: Sync,
{
    dual_reduce(subdomains, lambda, threads, Subdomain::local_solve)
}

/// The lumped FETI-DP preconditioner, `sum_s B_s K_dd,s B_s^T` — the same
/// reduction as `F` but with each subdomain's raw `K_dd,s` in place of its
/// inverse, trading a weaker convergence bound than the Dirichlet
/// preconditioner for a local matrix-vector product instead of a local
/// solve. Kept for the pathologically-small-subdomain case where Dirichlet's
/// extra setup cost doesn't pay for itself — not currently wired into
/// `projected_pcg` as the default (Dirichlet is), so only reachable from
/// tests without a runtime choice exposed yet.
#[allow(dead_code)]
pub(crate) fn dual_precondition<B>(
    subdomains: &[Subdomain<B>],
    lambda: &Vector,
    threads: usize,
) -> Vector
where
    B: Sync,
{
    dual_reduce(subdomains, lambda, threads, Subdomain::local_apply)
}

/// The Dirichlet FETI-DP preconditioner, `sum_s B_b,s S_s B_b,s^T` — the
/// theoretically optimal (near mesh-independent) choice, trading
/// `dual_precondition`'s cheap local matvec for one interior solve and two
/// rectangular matvecs per subdomain per application (`S_s` is never formed,
/// see `DirichletLocal`).
pub(crate) fn dual_precondition_dirichlet<B>(
    subdomains: &[Subdomain<B>],
    lambda: &Vector,
    threads: usize,
) -> Vector
where
    B: Sync,
{
    dual_reduce(
        subdomains,
        lambda,
        threads,
        Subdomain::local_dirichlet_apply,
    )
}

/// `C^T . lambda`, scattered into the global corner-DOF vector, where
/// `C = sum_s B_s K_dd,s^-1 K_dp,s = sum_s B_s . dual_map_s`.
fn coupling_transpose<B>(
    subdomains: &[Subdomain<B>],
    lambda: &Vector,
    num_corner_dofs: usize,
) -> Vector {
    subdomains
        .iter()
        .fold(Vector::zero(num_corner_dofs), |mut sum, subdomain| {
            let rhs = subdomain
                .interface()
                .apply_transpose(lambda, subdomain.num_local());
            let rhs_dual = subdomain.dual_rhs(&rhs);
            let contribution = &subdomain.dual_map().transpose() * &rhs_dual;
            subdomain
                .primal_global()
                .iter()
                .zip(contribution.iter())
                .for_each(|(&global, &value)| sum[global] += value);
            sum
        })
}

/// `C . v`, a global corner-DOF vector mapped into multiplier space.
fn coupling<B>(subdomains: &[Subdomain<B>], v: &Vector, num_multipliers: usize) -> Vector {
    subdomains
        .iter()
        .fold(Vector::zero(num_multipliers), |sum, subdomain| {
            let v_local: Vector = subdomain
                .primal_global()
                .iter()
                .map(|&global| v[global])
                .collect();
            let dual_contribution = subdomain.dual_map() * &v_local;
            let local = subdomain.scatter_dual(&dual_contribution);
            sum + subdomain.interface().apply(&local, num_multipliers)
        })
}

/// The action of the augmented dual operator `F + C S_pp^-1 C^T` — this, not
/// bare `F`, is what the coarse (corner) problem's elimination actually
/// leaves behind on the multiplier system; it carries the coarse-grid
/// correction into the dual problem, replacing classical FETI's separate
/// null-space projection step.
pub(crate) fn dual_operator<B>(
    subdomains: &[Subdomain<B>],
    lambda: &Vector,
    coarse: &Coarse,
    threads: usize,
) -> Vector
where
    B: Sync,
{
    let ct_lambda = coupling_transpose(subdomains, lambda, coarse.len());
    let coarse_solved = coarse.solve(&ct_lambda);
    dual_action(subdomains, lambda, threads) + coupling(subdomains, &coarse_solved, lambda.len())
}
