#![allow(dead_code)]

pub(crate) mod dual_primal;

use crate::math::{Scalar, Tensor, Vector, sparse::SparseSolver};

pub(crate) trait LocalSupport {}

pub(crate) struct Interface {
    dofs: Vec<usize>,
    signs: Vec<Scalar>,
}

impl Interface {
    pub(crate) fn from_shared_dofs(dofs: Vec<usize>, signs: Vec<Scalar>) -> Self {
        Self { dofs, signs }
    }
    pub(crate) fn dofs(&self) -> &[usize] {
        &self.dofs
    }
    fn apply(&self, local: &Vector) -> Vector {
        self.dofs
            .iter()
            .zip(self.signs.iter())
            .map(|(&dof, &sign)| sign * local[dof])
            .collect()
    }
    fn apply_transpose(&self, lambda: &Vector, num_local: usize) -> Vector {
        let mut local = Vector::zero(num_local);
        self.dofs
            .iter()
            .zip(self.signs.iter())
            .zip(lambda.iter())
            .for_each(|((&dof, &sign), &value)| local[dof] += sign * value);
        local
    }
}

pub(crate) struct Subdomain<B> {
    blocks: B,
    interface: Interface,
    solver: SparseSolver,
}

impl<B> Subdomain<B>
where
    B: LocalSupport,
{
    pub(crate) fn new(blocks: B, interface: Interface, solver: SparseSolver) -> Self {
        Self {
            blocks,
            interface,
            solver,
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

pub(crate) fn dual_action<B>(_subdomains: &[Subdomain<B>], _lambda: &Vector) -> Vector
where
    B: LocalSupport,
{
    todo!("sum over subdomains of B_s K_s^+ B_s^T lambda")
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
