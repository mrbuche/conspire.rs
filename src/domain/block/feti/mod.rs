#![allow(dead_code)]

pub(crate) mod dual_primal;
pub(crate) mod interface;
#[cfg(test)]
mod test;

use crate::math::{
    LuDecomposition, Matrix, SquareMatrix, Tensor, Vector,
    optimize::{Krylov, KrylovError, Preconditioning},
};
use dual_primal::coarse;
use interface::Interface;

pub(crate) struct Subdomain<B> {
    blocks: B,
    interface: Interface,
    dual_factor: LuDecomposition,
    dual_dofs: Vec<usize>,
    num_local: usize,
    /// `K_dd^-1 K_dp`, from `condense()` — maps a corner (primal) solution to
    /// this subdomain's dual correction, and is what carries the coarse-grid
    /// coupling term into the dual operator.
    dual_map: Matrix,
    /// Each local primal DOF's position in the global corner-DOF vector.
    primal_global: Vec<usize>,
}

impl<B> Subdomain<B> {
    pub(crate) fn new(
        blocks: B,
        interface: Interface,
        dual_factor: LuDecomposition,
        dual_dofs: Vec<usize>,
        num_local: usize,
        dual_map: Matrix,
        primal_global: Vec<usize>,
    ) -> Self {
        Self {
            blocks,
            interface,
            dual_factor,
            dual_dofs,
            num_local,
            dual_map,
            primal_global,
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
}

/// The action of the FETI-DP dual operator `F = sum_s B_s K_dd,s^-1 B_s^T` on
/// a multiplier vector: matrix-free, one independent local solve per
/// subdomain plus a reduction, never assembled.
pub(crate) fn dual_action<B>(subdomains: &[Subdomain<B>], lambda: &Vector) -> Vector {
    let num_multipliers = lambda.len();
    subdomains
        .iter()
        .map(|subdomain| {
            let rhs = subdomain
                .interface
                .apply_transpose(lambda, subdomain.num_local);
            let local = subdomain.local_solve(&rhs);
            subdomain.interface.apply(&local, num_multipliers)
        })
        .fold(Vector::zero(num_multipliers), |sum, contribution| {
            sum + contribution
        })
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
    schur: &SquareMatrix,
) -> Vector {
    let num_corner_dofs = schur.len();
    let ct_lambda = coupling_transpose(subdomains, lambda, num_corner_dofs);
    let coarse_solved = coarse::solve(schur, &ct_lambda);
    dual_action(subdomains, lambda) + coupling(subdomains, &coarse_solved, lambda.len())
}

/// Solves the dual (interface) problem `(F + C S_pp^-1 C^T) . lambda = rhs`
/// by conjugate gradients — SPD given the corners are pinned, so this needs
/// no projection against a rigid-body null space the way plain FETI would.
pub(crate) fn projected_pcg<B>(
    subdomains: &[Subdomain<B>],
    schur: &SquareMatrix,
    rhs: &Vector,
) -> Result<Vector, KrylovError> {
    Krylov::default().solve_operator(
        |lambda| dual_operator(subdomains, lambda, schur),
        Preconditioning::None,
        rhs,
    )
}

pub(crate) fn primal_recovery<B>(_subdomains: &[Subdomain<B>], _lambda: &Vector) -> Vec<Vector> {
    todo!("u_s = K_s^+(f_s - B_s^T lambda) + R_s alpha_s")
}
