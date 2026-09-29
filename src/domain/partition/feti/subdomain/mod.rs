#[cfg(test)]
mod test;

use crate::domain::feti::interface::Interface;
use crate::math::{LuDecomposition, Matrix, SquareMatrix, Tensor, Vector};
use std::collections::HashSet;

/// One subdomain's local system, already condensed and interfaced.
///
/// `blocks` is whatever per-subdomain payload the caller hangs on it (kept
/// generic over `B`, e.g. `()` in tests); everything else here is FETI-DP's
/// own bookkeeping built from it: the interface (jump) operator, the
/// condensed dual stiffness and its factorization, and the primal (corner)
/// numbering that ties this subdomain into the global coarse problem.
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

/// A subdomain's part of the Dirichlet preconditioner.
///
/// `S_s = K_bb - K_bi K_ii^-1 K_ib`, the Schur complement of its interior
/// (never touched by a multiplier) dual dofs `i` onto its boundary (touched
/// by one) dual dofs `b`. Applied implicitly, as `K_bb x` minus one interior
/// solve and two rectangular matvecs, instead of being formed: the
/// preconditioner runs a few dozen times, and forming `S_s` costs one
/// interior solve per boundary dof.
pub(crate) struct DirichletLocal {
    boundary_dofs: Vec<usize>,
    k_bb: SquareMatrix,
    k_bi: Matrix,
    k_ib: Matrix,
    interior_factor: Option<LuDecomposition>,
}

impl DirichletLocal {
    /// Splits a subdomain's dual dofs into interior (never touched by a
    /// multiplier) and boundary (touched by at least one), and extracts the
    /// blocks the Dirichlet preconditioner applies, factorizing `K_ii`.
    /// Interior-interior is a principal submatrix of the (SPD, once corners
    /// are condensed out) `K_dd,s`, hence always itself non-singular, so the
    /// factorization can't fail the way a corner elimination could on a
    /// floating subdomain. `K_bi` and `K_ib` are both kept, so the
    /// application needs no transposed products.
    pub(crate) fn build(
        local_stiffness: &SquareMatrix,
        dual_dofs: &[usize],
        interface_dofs: &[usize],
    ) -> Self {
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
        Self {
            k_bi: block(&boundary, &interior),
            k_ib: block(&interior, &boundary),
            boundary_dofs: boundary,
            k_bb,
            interior_factor,
        }
    }
    pub(crate) fn boundary_dofs(&self) -> &[usize] {
        &self.boundary_dofs
    }
    pub(crate) fn apply(&self, x: &Vector) -> Vector {
        let direct = &self.k_bb * x;
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
    pub(crate) fn num_local(&self) -> usize {
        self.num_local
    }
    pub(crate) fn dual_map(&self) -> &Matrix {
        &self.dual_map
    }
    pub(crate) fn primal_global(&self) -> &[usize] {
        &self.primal_global
    }
    /// Restricts a full-local vector to the subdomain dual (non-corner) DOFs.
    pub(crate) fn dual_rhs(&self, full_local: &Vector) -> Vector {
        self.dual_dofs.iter().map(|&dof| full_local[dof]).collect()
    }
    /// Scatters a dual-DOF vector back to its raw positions in the
    /// subdomain's full local numbering, leaving corner positions zero.
    pub(crate) fn scatter_dual(&self, dual_vector: &Vector) -> Vector {
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
    pub(crate) fn local_solve(&self, rhs: &Vector) -> Vector {
        let solved = self.dual_factor.solve(&self.dual_rhs(rhs));
        self.scatter_dual(&solved)
    }
    /// Applies `K_dd` directly (no solve) to `rhs`'s dual-restricted part —
    /// the local step the lumped preconditioner uses in place of a local
    /// solve, since it only needs to be cheap, not the true local inverse.
    pub(crate) fn local_apply(&self, rhs: &Vector) -> Vector {
        let applied = &self.dual_stiffness * &self.dual_rhs(rhs);
        self.scatter_dual(&applied)
    }
    /// Restricts a full-local vector to this subdomain's boundary (Γ) DOFs.
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
    pub(crate) fn local_dirichlet_apply(&self, rhs: &Vector) -> Vector {
        let applied = self.dirichlet.apply(&self.boundary_rhs(rhs));
        self.scatter_boundary(&applied)
    }
    /// Scatters a primal (corner)-DOF vector back to its raw positions in
    /// the subdomain's full local numbering, leaving dual positions zero.
    pub(crate) fn scatter_primal(&self, primal_vector: &Vector) -> Vector {
        let mut local = Vector::zero(self.num_local);
        self.primal_dofs
            .iter()
            .zip(primal_vector.iter())
            .for_each(|(&dof, &value)| local[dof] = value);
        local
    }
    /// Gathers this subdomain's local primal DOFs from the global
    /// corner-DOF vector.
    pub(crate) fn gather_primal(&self, corner_solution: &Vector) -> Vector {
        self.primal_global
            .iter()
            .map(|&global| corner_solution[global])
            .collect()
    }
}
