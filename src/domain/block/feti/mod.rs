#![allow(dead_code)]

pub(crate) mod dual_primal;
pub(crate) mod interface;
#[cfg(test)]
mod test;

use crate::math::{
    LuDecomposition, Tensor, Vector,
    optimize::{Krylov, KrylovError, Preconditioning},
};
use interface::Interface;

pub(crate) struct Subdomain<B> {
    blocks: B,
    interface: Interface,
    dual_factor: LuDecomposition,
    dual_dofs: Vec<usize>,
    num_local: usize,
}

impl<B> Subdomain<B> {
    pub(crate) fn new(
        blocks: B,
        interface: Interface,
        dual_factor: LuDecomposition,
        dual_dofs: Vec<usize>,
        num_local: usize,
    ) -> Self {
        Self {
            blocks,
            interface,
            dual_factor,
            dual_dofs,
            num_local,
        }
    }
    pub(crate) fn blocks(&self) -> &B {
        &self.blocks
    }
    pub(crate) fn interface(&self) -> &Interface {
        &self.interface
    }
    /// Solves `K_dd . x = rhs` restricted to this subdomain's dual (non-corner)
    /// DOFs, at their raw positions in the subdomain's full local numbering —
    /// the corner DOFs are never touched here, since they are pinned globally
    /// continuous and handled by the coarse problem instead. Non-singular by
    /// construction: pinning the corners is exactly what removes a floating
    /// subdomain's rigid-body modes from `K_dd`.
    fn local_solve(&self, rhs: &Vector) -> Vector {
        let rhs_dual: Vector = self.dual_dofs.iter().map(|&dof| rhs[dof]).collect();
        let solved = self.dual_factor.solve(&rhs_dual);
        let mut local = Vector::zero(self.num_local);
        self.dual_dofs
            .iter()
            .zip(solved.iter())
            .for_each(|(&dof, &value)| local[dof] = value);
        local
    }
}

/// The action of the FETI dual operator `F = sum_s B_s K_dd,s^-1 B_s^T` on a
/// multiplier vector: matrix-free, one independent local solve per subdomain
/// plus a reduction, never assembled.
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

/// Solves the dual (interface) problem `F . lambda = rhs` by conjugate
/// gradients — `F` is SPD given the corners are pinned, so this needs no
/// projection against a rigid-body null space the way plain FETI would.
pub(crate) fn projected_pcg<B>(
    subdomains: &[Subdomain<B>],
    rhs: &Vector,
) -> Result<Vector, KrylovError> {
    Krylov::default().solve_operator(
        |lambda| dual_action(subdomains, lambda),
        Preconditioning::None,
        rhs,
    )
}

pub(crate) fn primal_recovery<B>(_subdomains: &[Subdomain<B>], _lambda: &Vector) -> Vec<Vector> {
    todo!("u_s = K_s^+(f_s - B_s^T lambda) + R_s alpha_s")
}
