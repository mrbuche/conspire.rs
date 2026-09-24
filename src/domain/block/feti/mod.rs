#![allow(dead_code)]

#[cfg(feature = "fem")]
pub(crate) mod assemble;
pub(crate) mod dual_primal;
#[cfg(all(test, feature = "fem"))]
mod hex_test;
pub(crate) mod interface;
#[cfg(test)]
mod test;

use crate::math::{
    LuDecomposition, Matrix, Scalar, SquareMatrix, Tensor, Vector,
    optimize::{Krylov, KrylovError},
};
#[cfg(feature = "fem")]
use dual_primal::coarse;
use dual_primal::coarse::Coarse;
use interface::Interface;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::{available_parallelism, scope};
use std::time::Duration;
#[cfg(feature = "fem")]
use std::time::Instant;

/// Below this many subdomains, `dual_reduce` stays on the serial path —
/// thread-spawn overhead can otherwise exceed the per-subdomain work itself
/// (the same failure mode found and documented for parallel FEM assembly in
/// `[[parallel_assembly_plan]]`: "forces ~1ms of work can't pay per-call
/// spawn"). This threshold is a placeholder, not a benchmarked value — no
/// real multi-subdomain FETI-DP problem exists yet to tune it against.
const PARALLEL_THRESHOLD: usize = 4;

/// Most threads any parallel stage of the solve uses, setup and PCG alike.
const THREADS: usize = 4;

fn thread_count() -> usize {
    THREADS.min(available_parallelism().map_or(1, |threads| threads.get()))
}

/// `items.iter().map(f).collect()` spread over up to `THREADS` threads.
/// Threads pull the next unclaimed item, not a fixed chunk, because
/// subdomains cost different amounts (an interior subdomain carries more dual
/// dofs than a corner one), and results come back in item order.
fn parallel_map<T, R>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R>
where
    T: Sync,
    R: Send,
{
    let threads = thread_count().min(items.len());
    if threads <= 1 {
        return items.iter().map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let mut results: Vec<(usize, R)> = scope(|scope| {
        (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= items.len() {
                            break done;
                        }
                        done.push((index, f(&items[index])));
                    }
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flat_map(|handle| handle.join().expect("setup thread panicked"))
            .collect()
    });
    results.sort_unstable_by_key(|&(index, _)| index);
    results.into_iter().map(|(_, result)| result).collect()
}

#[cfg(feature = "fem")]
use crate::{
    constitutive::solid::hyperelastic::Hyperelastic,
    fem::{
        NodalCoordinates,
        block::{
            Block,
            element::{FiniteElementError, solid::hyperelastic::HyperelasticFiniteElement},
        },
    },
};

/// `matrix . vector` by reference — the preconditioner applies a dense local
/// matrix on every PCG iteration, so cloning it per application is a cost of
/// its own.
fn matvec(matrix: &SquareMatrix, vector: &Vector) -> Vector {
    matrix.iter().map(|row| row * vector).collect()
}

pub(crate) struct Subdomain<B> {
    blocks: B,
    interface: Interface,
    /// The dual (non-corner) block of the local stiffness, `K_dd,s`, kept
    /// alongside its factorization for the lumped preconditioner's local
    /// apply, which needs `K_dd,s` itself rather than its inverse.
    dual_stiffness: SquareMatrix,
    dual_factor: LuDecomposition,
    dual_dofs: Vec<usize>,
    num_local: usize,
    /// `K_dd^-1 K_dp`, from `condense()` — maps a corner (primal) solution to
    /// this subdomain's dual correction, and is what carries the coarse-grid
    /// coupling term into the dual operator.
    dual_map: Matrix,
    /// Each local primal (corner) DOF's raw position in this subdomain's full
    /// local numbering.
    primal_dofs: Vec<usize>,
    /// Each local primal DOF's position in the global corner-DOF vector,
    /// parallel to `primal_dofs`.
    primal_global: Vec<usize>,
    /// What the Dirichlet preconditioner needs from this subdomain.
    dirichlet: DirichletLocal,
}

/// A subdomain's part of the Dirichlet preconditioner,
/// `S_s = K_bb - K_bi K_ii^-1 K_ib`, the Schur complement of its interior
/// (never touched by a multiplier) dual dofs `i` onto its boundary (touched
/// by one) dual dofs `b`. `S_s` is applied implicitly, as `K_bb x` minus one
/// interior solve and two rectangular matvecs, instead of being formed: the
/// preconditioner runs a few dozen times, and forming `S_s` costs one interior
/// solve per boundary dof.
pub(crate) struct DirichletLocal {
    /// Raw local positions of the boundary dofs, in the row order of the
    /// blocks below.
    boundary_dofs: Vec<usize>,
    k_bb: SquareMatrix,
    k_bi: Matrix,
    k_ib: Matrix,
    /// `None` when the subdomain has no interior dofs.
    interior_factor: Option<LuDecomposition>,
}

impl DirichletLocal {
    pub(crate) fn boundary_dofs(&self) -> &[usize] {
        &self.boundary_dofs
    }
    /// `S_s . x` for `x` given on the boundary dofs.
    pub(crate) fn apply(&self, x: &Vector) -> Vector {
        let direct = matvec(&self.k_bb, x);
        match &self.interior_factor {
            None => direct,
            Some(factor) => {
                let interior = factor.solve(&(&self.k_ib * x));
                direct - &self.k_bi * &interior
            }
        }
    }
}

impl<B> Subdomain<B> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        blocks: B,
        interface: Interface,
        dual_stiffness: SquareMatrix,
        dual_factor: LuDecomposition,
        dual_dofs: Vec<usize>,
        num_local: usize,
        dual_map: Matrix,
        primal_dofs: Vec<usize>,
        primal_global: Vec<usize>,
        dirichlet: DirichletLocal,
    ) -> Self {
        Self {
            blocks,
            interface,
            dual_stiffness,
            dual_factor,
            dual_dofs,
            num_local,
            dual_map,
            primal_dofs,
            primal_global,
            dirichlet,
        }
    }
    pub(crate) fn blocks(&self) -> &B {
        &self.blocks
    }
    pub(crate) fn interface(&self) -> &Interface {
        &self.interface
    }
    /// Restricts a full-local vector to this subdomain's dual (non-corner)
    /// DOFs.
    fn dual_rhs(&self, full_local: &Vector) -> Vector {
        self.dual_dofs.iter().map(|&dof| full_local[dof]).collect()
    }
    /// Scatters a dual-DOF vector back to its raw positions in the
    /// subdomain's full local numbering, leaving corner positions zero.
    fn scatter_dual(&self, dual_vector: &Vector) -> Vector {
        let mut local = Vector::zero(self.num_local);
        self.dual_dofs
            .iter()
            .zip(dual_vector.iter())
            .for_each(|(&dof, &value)| local[dof] = value);
        local
    }
    /// Solves `K_dd . x = rhs` restricted to this subdomain's dual (non-corner)
    /// DOFs, at their raw positions in the subdomain's full local numbering —
    /// the corner DOFs are never touched here, since they are pinned globally
    /// continuous and handled by the coarse problem instead. Non-singular by
    /// construction: pinning the corners is exactly what removes a floating
    /// subdomain's rigid-body modes from `K_dd`.
    fn local_solve(&self, rhs: &Vector) -> Vector {
        let solved = self.dual_factor.solve(&self.dual_rhs(rhs));
        self.scatter_dual(&solved)
    }
    /// Applies `K_dd` directly (no solve) to `rhs`'s dual-restricted part —
    /// the local step the lumped preconditioner uses in place of a local
    /// solve, since it only needs to be cheap, not the true local inverse.
    fn local_apply(&self, rhs: &Vector) -> Vector {
        let applied = matvec(&self.dual_stiffness, &self.dual_rhs(rhs));
        self.scatter_dual(&applied)
    }
    /// Restricts a full-local vector to this subdomain's boundary (Γ) dofs.
    fn boundary_rhs(&self, full_local: &Vector) -> Vector {
        self.dirichlet
            .boundary_dofs()
            .iter()
            .map(|&dof| full_local[dof])
            .collect()
    }
    /// Scatters a boundary-dof vector back to its raw positions in the
    /// subdomain's full local numbering, leaving every other position zero.
    fn scatter_boundary(&self, boundary_vector: &Vector) -> Vector {
        let mut local = Vector::zero(self.num_local);
        self.dirichlet
            .boundary_dofs()
            .iter()
            .zip(boundary_vector.iter())
            .for_each(|(&dof, &value)| local[dof] = value);
        local
    }
    /// Applies the Dirichlet preconditioner's local contribution `S_s` to
    /// `rhs`'s boundary-restricted part — the local step the Dirichlet
    /// preconditioner uses in place of `local_apply`'s raw `K_dd`, giving the
    /// near mesh-independent condition number bound.
    fn local_dirichlet_apply(&self, rhs: &Vector) -> Vector {
        let applied = self.dirichlet.apply(&self.boundary_rhs(rhs));
        self.scatter_boundary(&applied)
    }
    /// Scatters a primal (corner)-DOF vector back to its raw positions in
    /// the subdomain's full local numbering, leaving dual positions zero.
    fn scatter_primal(&self, primal_vector: &Vector) -> Vector {
        let mut local = Vector::zero(self.num_local);
        self.primal_dofs
            .iter()
            .zip(primal_vector.iter())
            .for_each(|(&dof, &value)| local[dof] = value);
        local
    }
    /// Gathers this subdomain's local primal DOFs from the global
    /// corner-DOF vector.
    fn gather_primal(&self, corner_solution: &Vector) -> Vector {
        self.primal_global
            .iter()
            .map(|&global| corner_solution[global])
            .collect()
    }
}

/// Reduces a per-subdomain local step (`local_solve` for the dual operator,
/// `local_apply` for the lumped preconditioner) into a multiplier-space
/// vector: matrix-free, one independent local step per subdomain plus a
/// reduction, never assembled.
fn dual_reduce<B>(
    subdomains: &[Subdomain<B>],
    lambda: &Vector,
    local_step: impl Fn(&Subdomain<B>, &Vector) -> Vector + Sync,
) -> Vector
where
    B: Sync,
{
    let num_multipliers = lambda.len();
    // All captures here (`lambda`, `local_step`, `num_multipliers`) are
    // references or Copy, so this closure is itself Copy — sharing it across
    // spawned threads below is just copying a handful of references, not
    // moving anything that can only live in one place.
    let reduce = |chunk: &[Subdomain<B>]| {
        chunk
            .iter()
            .map(|subdomain| {
                let rhs = subdomain
                    .interface
                    .apply_transpose(lambda, subdomain.num_local);
                let local = local_step(subdomain, &rhs);
                subdomain.interface.apply(&local, num_multipliers)
            })
            .fold(Vector::zero(num_multipliers), |sum, contribution| {
                sum + contribution
            })
    };
    if subdomains.len() < PARALLEL_THRESHOLD {
        return reduce(subdomains);
    }
    // Each subdomain's local step is fully independent (no shared mutable
    // state during the parallel phase), so this is the simple map-then-
    // reduce case from [[parallel_assembly_plan]] (option 2 there, "per-
    // thread accumulators + reduction") rather than anything needing
    // coloring or row-gather — those solve write conflicts scattering into
    // one shared structure, which doesn't arise here since each thread only
    // ever produces its own small partial-sum `Vector`.
    let chunk_size = subdomains.len().div_ceil(thread_count()).max(1);
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
pub(crate) fn dual_action<B>(subdomains: &[Subdomain<B>], lambda: &Vector) -> Vector
where
    B: Sync,
{
    dual_reduce(subdomains, lambda, Subdomain::local_solve)
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
fn dual_precondition<B>(subdomains: &[Subdomain<B>], lambda: &Vector) -> Vector
where
    B: Sync,
{
    dual_reduce(subdomains, lambda, Subdomain::local_apply)
}

/// The Dirichlet FETI-DP preconditioner, `sum_s B_b,s S_s B_b,s^T` — the
/// theoretically optimal (near mesh-independent) choice, trading
/// `dual_precondition`'s cheap local matvec for one interior solve and two
/// rectangular matvecs per subdomain per application (`S_s` is never formed,
/// see `DirichletLocal`).
fn dual_precondition_dirichlet<B>(subdomains: &[Subdomain<B>], lambda: &Vector) -> Vector
where
    B: Sync,
{
    dual_reduce(subdomains, lambda, Subdomain::local_dirichlet_apply)
}

/// Splits a subdomain's dual dofs into interior (never touched by a
/// multiplier) and boundary (touched by at least one), and extracts the
/// blocks the Dirichlet preconditioner applies, factorizing `K_ii`.
/// Interior-interior is a principal submatrix of the (SPD, once corners are
/// condensed out) `K_dd,s`, hence always itself non-singular, so the
/// factorization can't fail the way a corner elimination could on a floating
/// subdomain. `K_bi` and `K_ib` are both kept, so the application needs no
/// transposed products.
fn dirichlet_local(
    local_stiffness: &SquareMatrix,
    dual_dofs: &[usize],
    interface_dofs: &[usize],
) -> DirichletLocal {
    let on_interface: HashSet<usize> = interface_dofs.iter().copied().collect();
    let boundary: Vec<usize> = dual_dofs
        .iter()
        .copied()
        .filter(|dof| on_interface.contains(dof))
        .collect();
    let interior: Vec<usize> = dual_dofs
        .iter()
        .copied()
        .filter(|dof| !on_interface.contains(dof))
        .collect();
    let block = |rows: &[usize], columns: &[usize]| -> Matrix {
        rows.iter()
            .map(|&row| {
                columns
                    .iter()
                    .map(|&column| local_stiffness[row][column])
                    .collect()
            })
            .collect()
    };
    let k_bb: SquareMatrix = boundary
        .iter()
        .map(|&row| {
            boundary
                .iter()
                .map(|&column| local_stiffness[row][column])
                .collect()
        })
        .collect();
    let interior_factor = if interior.is_empty() {
        None
    } else {
        let k_ii: SquareMatrix = interior
            .iter()
            .map(|&row| {
                interior
                    .iter()
                    .map(|&column| local_stiffness[row][column])
                    .collect()
            })
            .collect();
        Some(
            k_ii.factorize_lu()
                .expect("K_ii is singular, but it is a principal block of a non-singular K_dd"),
        )
    };
    DirichletLocal {
        k_bi: block(&boundary, &interior),
        k_ib: block(&interior, &boundary),
        boundary_dofs: boundary,
        k_bb,
        interior_factor,
    }
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
                .interface
                .apply_transpose(lambda, subdomain.num_local);
            let rhs_dual = subdomain.dual_rhs(&rhs);
            let contribution = &subdomain.dual_map.transpose() * &rhs_dual;
            subdomain
                .primal_global
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
                .primal_global
                .iter()
                .map(|&global| v[global])
                .collect();
            let dual_contribution = &subdomain.dual_map * &v_local;
            let local = subdomain.scatter_dual(&dual_contribution);
            sum + subdomain.interface.apply(&local, num_multipliers)
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
) -> Vector
where
    B: Sync,
{
    let ct_lambda = coupling_transpose(subdomains, lambda, coarse.len());
    let coarse_solved = coarse.solve(&ct_lambda);
    dual_action(subdomains, lambda) + coupling(subdomains, &coarse_solved, lambda.len())
}

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
    projected_pcg_counted(
        subdomains,
        coarse,
        rhs,
        Preconditioner::Dirichlet,
        Krylov::default().rel_tol,
    )
    .map(|(lambda, _)| lambda)
}

/// Wall time of each stage of a `solve_with`, and the number of dual
/// operator applications the PCG took.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SolveStats {
    pub(crate) applications: usize,
    pub(crate) assemble: Duration,
    pub(crate) condense: Duration,
    pub(crate) subdomains: Duration,
    pub(crate) coarse: Duration,
    pub(crate) pcg: Duration,
    pub(crate) recover: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Preconditioner {
    Lumped,
    Dirichlet,
}

/// `projected_pcg` with the preconditioner chosen explicitly, also returning
/// how many times the dual operator was applied (one per PCG iteration).
pub(crate) fn projected_pcg_counted<B>(
    subdomains: &[Subdomain<B>],
    coarse: &Coarse,
    rhs: &Vector,
    preconditioner: Preconditioner,
    rel_tol: Scalar,
) -> Result<(Vector, usize), KrylovError>
where
    B: Sync,
{
    let mut applications = 0;
    let lambda = Krylov {
        rel_tol,
        ..Krylov::default()
    }
    .solve_operator(
        |lambda| {
            applications += 1;
            dual_operator(subdomains, lambda, coarse)
        },
        |lambda: &Vector| match preconditioner {
            Preconditioner::Lumped => dual_precondition(subdomains, lambda),
            Preconditioner::Dirichlet => dual_precondition_dirichlet(subdomains, lambda),
        },
        rhs,
    )?;
    Ok((lambda, applications))
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
                .interface
                .apply_transpose(lambda, subdomain.num_local);
            let combined_rhs: Vector = local_force
                .iter()
                .zip(multiplier_rhs.iter())
                .map(|(&f, &m)| f - m)
                .collect();
            let dual_solution = subdomain.local_solve(&combined_rhs);
            let primal_local = subdomain.gather_primal(corner_solution);
            let coupling_correction =
                subdomain.scatter_dual(&(&subdomain.dual_map * &primal_local));
            dual_solution - coupling_correction + subdomain.scatter_primal(&primal_local)
        })
        .collect()
}

/// `sum_s B_s K_dd,s^-1 f_d,s` — the dual (interface) problem's right-hand
/// side contribution from the forces alone, before the coarse-coupling
/// correction `solve` applies on top.
#[cfg(feature = "fem")]
fn rhs_from_forces<B>(
    subdomains: &[Subdomain<B>],
    local_forces: &[Vector],
    num_multipliers: usize,
) -> Vector {
    subdomains.iter().zip(local_forces.iter()).fold(
        Vector::zero(num_multipliers),
        |sum, (subdomain, force)| {
            let local = subdomain.local_solve(force);
            sum + subdomain.interface.apply(&local, num_multipliers)
        },
    )
}

/// Errors a full FETI-DP solve can hit: either extracting a subdomain's
/// local stiffness/force from the real `Block` fails, or the dual PCG does.
#[cfg(feature = "fem")]
pub(crate) enum SolveError {
    Element(FiniteElementError),
    Krylov(KrylovError),
}

#[cfg(feature = "fem")]
impl From<FiniteElementError> for SolveError {
    fn from(error: FiniteElementError) -> Self {
        Self::Element(error)
    }
}

#[cfg(feature = "fem")]
impl From<KrylovError> for SolveError {
    fn from(error: KrylovError) -> Self {
        Self::Krylov(error)
    }
}

/// Solves a real FEM `Block` by FETI-DP, end to end: extracts each
/// subdomain's local stiffness/force, condenses the dual DOFs, assembles and
/// solves the coarse corner problem, runs the Dirichlet-preconditioned dual PCG
/// with the coarse-coupling correction, recovers each subdomain's local
/// solution, and scatters everything back into one global nodal vector.
///
/// Hyperelastic models only: FETI-DP as built needs a symmetric tangent
/// (`C^T = dual_map^T`, and the dual PCG is conjugate gradients), and an
/// elastic-only model such as `AlmansiHamelEulerian` has an asymmetric one
/// away from zero deformation — measured 1e-2 asymmetry gave a 1e-4 error
/// against a dense global solve, where `NeoHookean` matched to 1e-12.
///
/// `partition` is the mesh decomposition, assumed given (an external
/// decomposer's job, not this solver's). Corners are chosen by
/// [`CornerSelection::from_partition`]'s standard heuristic. `boundary_conditions`
/// pins whichever (global node, component) DOFs are externally supported —
/// without at least enough of them to remove every global rigid-body mode,
/// the assembled coarse problem is singular and `solve` fails; corner
/// condensation alone only removes each subdomain's own LOCAL floating
/// modes, never a mode that moves the whole structure together.
#[cfg(feature = "fem")]
#[allow(clippy::type_complexity)]
pub(crate) fn solve<C, F, const G: usize, const M: usize, const N: usize, const P: usize>(
    block: &Block<C, F, G, M, N, P>,
    nodal_coordinates: &NodalCoordinates<3>,
    partition: &crate::geometry::mesh::Partition,
    boundary_conditions: &dual_primal::BoundaryConditions,
    dimension: usize,
) -> Result<Vector, SolveError>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, M, N, P>,
{
    solve_with(
        block,
        nodal_coordinates,
        partition,
        boundary_conditions,
        dimension,
        Preconditioner::Dirichlet,
        Krylov::default().rel_tol,
    )
    .map(|(solution, _)| solution)
}

/// `solve` with the dual PCG preconditioner chosen explicitly, also
/// returning the number of dual operator applications the PCG took.
#[cfg(feature = "fem")]
#[allow(clippy::type_complexity)]
pub(crate) fn solve_with<C, F, const G: usize, const M: usize, const N: usize, const P: usize>(
    block: &Block<C, F, G, M, N, P>,
    nodal_coordinates: &NodalCoordinates<3>,
    partition: &crate::geometry::mesh::Partition,
    boundary_conditions: &dual_primal::BoundaryConditions,
    dimension: usize,
    preconditioner: Preconditioner,
    rel_tol: Scalar,
) -> Result<(Vector, SolveStats), SolveError>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, M, N, P>,
{
    let mut stats = SolveStats::default();
    let corners = dual_primal::CornerSelection::from_partition(partition);
    let (interfaces, num_multipliers) = interface::build_interfaces(partition, &corners, dimension);
    let (splits, corner_dofs) =
        dual_primal::build_splits(partition, &corners, boundary_conditions, dimension);
    let subdomain_nodes = partition.parts_nodes();
    let clock = Instant::now();
    let (local_stiffnesses, local_forces): (Vec<SquareMatrix>, Vec<Vector>) = subdomain_nodes
        .iter()
        .map(|nodes| assemble::local_stiffness_and_force(block, nodal_coordinates, nodes))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();
    stats.assemble = clock.elapsed();
    let indices: Vec<usize> = (0..subdomain_nodes.len()).collect();
    let clock = Instant::now();
    let condensed: Vec<_> = parallel_map(&indices, |&index| {
        dual_primal::condense::condense(
            &local_stiffnesses[index],
            &local_forces[index],
            splits[index].primal(),
            splits[index].dual(),
        )
    });
    stats.condense = clock.elapsed();
    let clock = Instant::now();
    let (schur, reduced_force) = coarse::assemble(&condensed, &splits, &corner_dofs);
    let coarse_problem = Coarse::new(schur);
    stats.coarse = clock.elapsed();
    let clock = Instant::now();
    let locals = parallel_map(&indices, |&index| {
        let stiffness = &local_stiffnesses[index];
        let dual_dofs = splits[index].dual().to_vec();
        let dual_stiffness: SquareMatrix = dual_dofs
            .iter()
            .map(|&row| dual_dofs.iter().map(|&col| stiffness[row][col]).collect())
            .collect();
        let dual_factor = dual_stiffness
            .factorize_lu()
            .expect("K_dd is singular, but corners should make every subdomain non-singular");
        let dirichlet = dirichlet_local(stiffness, &dual_dofs, interfaces[index].dofs());
        (dual_dofs, dual_stiffness, dual_factor, dirichlet)
    });
    let subdomains: Vec<Subdomain<()>> = interfaces
        .into_iter()
        .zip(locals)
        .zip(splits.iter())
        .zip(condensed.iter())
        .zip(subdomain_nodes.iter())
        .map(|((((interface, local), split), condensed), nodes)| {
            let (dual_dofs, dual_stiffness, dual_factor, dirichlet) = local;
            Subdomain::new(
                (),
                interface,
                dual_stiffness,
                dual_factor,
                dual_dofs,
                nodes.len() * dimension,
                condensed.dual_map.clone(),
                split.primal().to_vec(),
                split.primal_global().to_vec(),
                dirichlet,
            )
        })
        .collect();
    stats.subdomains = clock.elapsed();
    let clock = Instant::now();
    let rhs = rhs_from_forces(&subdomains, &local_forces, num_multipliers)
        - coupling(
            &subdomains,
            &coarse_problem.solve(&reduced_force),
            num_multipliers,
        );
    stats.coarse += clock.elapsed();
    let clock = Instant::now();
    let (lambda, applications) =
        projected_pcg_counted(&subdomains, &coarse_problem, &rhs, preconditioner, rel_tol)?;
    stats.applications = applications;
    stats.pcg = clock.elapsed();
    let clock = Instant::now();
    let ct_lambda = coupling_transpose(&subdomains, &lambda, coarse_problem.len());
    let corner_solution = coarse_problem.solve(&(reduced_force + ct_lambda));
    let recovered = primal_recovery(&subdomains, &local_forces, &corner_solution, &lambda);
    let mut global = Vector::zero(nodal_coordinates.len() * dimension);
    subdomain_nodes
        .iter()
        .zip(recovered.iter())
        .for_each(|(nodes, local_solution)| {
            nodes.iter().enumerate().for_each(|(local, &node)| {
                (0..dimension).for_each(|component| {
                    global[dimension * node + component] =
                        local_solution[dimension * local + component]
                })
            })
        });
    stats.recover = clock.elapsed();
    Ok((global, stats))
}
