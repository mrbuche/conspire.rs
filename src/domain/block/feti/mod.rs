#![allow(dead_code)]

pub(crate) mod dual_primal;
pub(crate) mod interface;

use crate::math::{Tensor, Vector, sparse::SparseSolver};
use interface::Interface;

pub(crate) trait LocalSupport {}

pub(crate) struct Subdomain<B> {
    blocks: B,
    interface: Interface,
    solver: SparseSolver,
    num_local: usize,
}

impl<B> Subdomain<B>
where
    B: LocalSupport,
{
    pub(crate) fn new(
        blocks: B,
        interface: Interface,
        solver: SparseSolver,
        num_local: usize,
    ) -> Self {
        Self {
            blocks,
            interface,
            solver,
            num_local,
        }
    }
    pub(crate) fn blocks(&self) -> &B {
        &self.blocks
    }
    pub(crate) fn interface(&self) -> &Interface {
        &self.interface
    }
    fn local_solve(&self, _rhs: &Vector) -> Vector {
        todo!("local K_s solve, singular for floating subdomains")
    }
}

pub(crate) fn dual_action<B>(subdomains: &[Subdomain<B>], lambda: &Vector) -> Vector
where
    B: LocalSupport,
{
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

pub(crate) fn projected_pcg<B>(_subdomains: &[Subdomain<B>], _rhs: &Vector) -> Vector
where
    B: LocalSupport,
{
    todo!("PCG on the dual interface problem, projected against the coarse space")
}

pub(crate) fn primal_recovery<B>(_subdomains: &[Subdomain<B>], _lambda: &Vector) -> Vec<Vector>
where
    B: LocalSupport,
{
    todo!("u_s = K_s^+(f_s - B_s^T lambda) + R_s alpha_s")
}
