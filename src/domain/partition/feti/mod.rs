#![allow(dead_code)]

#[cfg(test)]
mod test;

pub(crate) mod block;
pub(crate) mod dual;
pub(crate) mod dual_primal;
pub(crate) mod interface;
pub(crate) mod parallel;
pub(crate) mod pcg;
pub(crate) mod solid;
pub(crate) mod subdomain;
#[cfg(feature = "fem")]
pub(crate) mod thermal;

pub use block::element::{DecomposableElements, ElementSystems};
#[cfg(feature = "fem")]
pub use block::solve::SolveError;
#[cfg(feature = "fem")]
pub use dual_primal::BoundaryConditions;
#[cfg(feature = "fem")]
pub use pcg::Preconditioner;

#[cfg(feature = "fem")]
use crate::domain::NodalCoordinates;
#[cfg(feature = "fem")]
use crate::{
    geometry::mesh::Partition,
    math::{
        Matrix, Scalar, Tensor, Vector,
        optimize::{Krylov, KrylovMethod, LinearSolver},
    },
};
#[cfg(feature = "fem")]
use block::solve::solve_local_systems;

pub(crate) const THREADS: usize = 1;

/// The dual solve by GMRES, for a nonsymmetric dual operator, restarting
/// only after a generous 100 iterations since the solve converges in tens.
///
/// Not the default: conjugate gradients also refuses a singular subdomain
/// through its positive-definiteness check, which GMRES has no way to do.
#[cfg(feature = "fem")]
pub const GMRES: KrylovMethod = KrylovMethod::Gmres(100);

/// How the subdomains are tied together.
///
/// Classical FETI ties every interface DOF with a Lagrange multiplier, so
/// there are no corners and every subdomain is floating unless boundary
/// conditions pin it. A floating subdomain's stiffness is singular along its
/// rigid-body modes, so its local solve is a generalized inverse and the dual
/// solve is projected against those modes.
///
/// A subdomain's rigid-body modes are an exact kernel of its tangent only
/// where it carries no stress, which a subdomain cut out of a stressed body
/// does at its interface. Classical FETI takes them as the kernel anyway, so
/// unlike FETI-DP it is inexact for a geometrically nonlinear tangent, by an
/// error that grows with the strain: 2e-3 of the solution at the strains of
/// the tests, and 1e-2 of the strain at most. Inside Newton's method the
/// solution is still the right one, but each step is approximate. The tangent
/// must also be symmetric.
///
/// Its Dirichlet preconditioner is scaled by multiplicity, which it needs:
/// unscaled it takes many times more iterations, growing with the number of
/// subdomains. FETI-DP takes about the same either way, so it is not scaled.
#[cfg(feature = "fem")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Formulation {
    /// No corner nodes.
    Classical,
    /// Corner nodes are primal.
    #[default]
    DualPrimal,
}

/// FETI-DP solver for the linearized systems of a decomposable block.
///
/// The block is split by a [`Partition`], and each subdomain is solved
/// independently, tied together through the corner DOFs and Lagrange
/// multipliers on the interface. Any elastic block is supported. The tangent
/// need not be symmetric, but then the dual solve must be [`GMRES`], since
/// conjugate gradients, the default, needs a symmetric positive definite dual
/// operator, and is what refuses a tangent that is not.
///
/// Boundary conditions are pinned DOFs, held at zero by elimination, and
/// linear constraints, enforced by multipliers, which may prescribe a nonzero
/// displacement. Between them, enough DOFs must be constrained to remove
/// every rigid-body mode of the whole block.
///
/// A constraint of one DOF is a row of the interface operator, held by a
/// multiplier there, with only as many of its nodes made corners as the
/// rigid-body modes of the subdomains and of the block call for. With such
/// rows the dual operator is no longer positive definite once the tangent is
/// stressed, so the dual solve is [`GMRES`] in place of conjugate gradients.
///
/// As the linear solver of a [`NewtonRaphson`](crate::math::optimize::NewtonRaphson),
/// it works with a fixed equality constraint, whose fixed DOFs are the pinned
/// ones, or a linear one, whose rows are the constraints.
#[cfg(feature = "fem")]
#[derive(Clone, Debug)]
pub struct Feti {
    pub formulation: Formulation,
    pub method: KrylovMethod,
    pub partition: Partition,
    pub preconditioner: Preconditioner,
    pub rel_tol: Scalar,
}

#[cfg(feature = "fem")]
impl Default for Feti {
    fn default() -> Self {
        Self {
            method: KrylovMethod::ConjugateGradients,
            formulation: Formulation::DualPrimal,
            partition: Partition::default(),
            preconditioner: Preconditioner::Dirichlet,
            rel_tol: Krylov::default().rel_tol,
        }
    }
}

#[cfg(feature = "fem")]
impl Feti {
    pub fn solve<B>(
        &self,
        block: &B,
        nodal_coordinates: &NodalCoordinates<3>,
        boundary_conditions: &BoundaryConditions,
    ) -> Result<Vector, SolveError>
    where
        B: DecomposableElements,
    {
        self.solve_constrained(block, nodal_coordinates, boundary_conditions)
            .map(|(solution, _)| solution)
    }
    /// Solves the linearized system, also giving the multiplier of each
    /// constraint of the boundary conditions, in the order they were added.
    ///
    /// The multipliers are the constraint forces, such that `K u + A^T mu = f`
    /// for the constraint matrix `A`, so `-mu` is the reaction of a
    /// prescribed displacement.
    pub fn solve_constrained<B>(
        &self,
        block: &B,
        nodal_coordinates: &NodalCoordinates<3>,
        boundary_conditions: &BoundaryConditions,
    ) -> Result<(Vector, Vector), SolveError>
    where
        B: DecomposableElements,
    {
        let systems = block.element_systems(nodal_coordinates)?;
        self.solve_systems(&systems, boundary_conditions)
    }
    fn solve_systems(
        &self,
        systems: &ElementSystems,
        boundary_conditions: &BoundaryConditions,
    ) -> Result<(Vector, Vector), SolveError> {
        let (stiffnesses, forces) = systems
            .subdomains(&self.partition)
            .map_err(SolveError::Partition)?;
        solve_local_systems(
            &self.partition,
            boundary_conditions,
            stiffnesses,
            forces,
            systems.positions(),
            self.preconditioner,
            self.rel_tol,
            self.method,
            self.formulation,
        )
    }
}

#[cfg(feature = "fem")]
impl LinearSolver for Feti {
    type Tangent = ElementSystems;
    fn solve(
        &self,
        tangent: ElementSystems,
        retained: &[usize],
        _residual: &Vector,
    ) -> Result<Vector, String> {
        let mut fixed = vec![true; 3 * tangent.positions().len()];
        retained.iter().for_each(|&dof| fixed[dof] = false);
        let boundary_conditions = BoundaryConditions::new(
            fixed
                .iter()
                .enumerate()
                .filter(|&(_, &pinned)| pinned)
                .map(|(dof, _)| (dof / 3, dof % 3))
                .collect(),
        );
        let solution = self
            .solve_systems(&tangent, &boundary_conditions)
            .map_err(|error| error.to_string())?
            .0;
        Ok(retained.iter().map(|&dof| solution[dof]).collect())
    }
    fn solve_constrained(
        &self,
        tangent: ElementSystems,
        constraint_matrix: &Matrix,
        multipliers: &Vector,
        residual: &Vector,
    ) -> Result<Vector, String> {
        let num_variables = residual.len() - constraint_matrix.len();
        let boundary_conditions = constraint_matrix.iter().enumerate().fold(
            BoundaryConditions::none(),
            |boundary_conditions, (row, coefficients)| {
                boundary_conditions.linear(
                    coefficients
                        .iter()
                        .enumerate()
                        .filter(|&(_, &coefficient)| coefficient != 0.0)
                        .map(|(dof, &coefficient)| (dof / 3, dof % 3, coefficient))
                        .collect(),
                    -residual[num_variables + row],
                )
            },
        );
        let (decrement, constraint_forces) = self
            .solve_systems(&tangent, &boundary_conditions)
            .map_err(|error| error.to_string())?;
        Ok(decrement
            .iter()
            .copied()
            .chain(
                multipliers
                    .iter()
                    .zip(constraint_forces.iter())
                    .map(|(multiplier, force)| multiplier - force),
            )
            .collect())
    }
}
