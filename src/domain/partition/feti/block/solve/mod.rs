#[cfg(test)]
mod test;

use super::{
    super::{
        THREADS,
        dual::{coupling, coupling_transpose, rhs_from_forces},
        dual_primal::{
            BoundaryConditions, CornerSelection, build_splits,
            coarse::{Coarse, CoarseSystem},
            condense::Condensed,
            rigid::removed_modes,
        },
        interface::build_interfaces,
        parallel::parallel_map,
        pcg::{Preconditioner, primal_recovery, projected_pcg_with},
        subdomain::{DirichletLocal, Subdomain},
    },
    {assemble::local_stiffness_and_force, element::positions},
};
use crate::{
    constitutive::solid::hyperelastic::Hyperelastic,
    fem::{
        NodalCoordinates,
        block::{
            Block,
            element::{FiniteElementError, solid::hyperelastic::HyperelasticFiniteElement},
        },
    },
    geometry::mesh::Partition,
    math::{
        Scalar, SquareMatrix, Style, StyledError, Vector,
        optimize::{Krylov, KrylovError},
        styled_error,
    },
};
use std::collections::HashSet;

/// Possible errors encountered when solving with FETI.
pub enum SolveError {
    /// Downstream error from a finite element.
    Element(FiniteElementError),
    /// Downstream error from the dual PCG.
    Krylov(KrylovError),
    /// A subdomain left with some rigid-body modes free.
    FloatingSubdomain { part: usize, removed: usize },
    /// A subdomain with some part of it still free to move.
    SingularSubdomain(usize),
    /// The assembled corner problem is singular.
    SingularCoarseProblem,
}

impl StyledError for SolveError {
    fn message(&self, style: &Style) -> String {
        match self {
            Self::Element(error) => error.message(style),
            Self::Krylov(error) => error.message(style),
            Self::FloatingSubdomain { part, removed } => {
                let (h, c) = (style.headline, style.frame);
                format!(
                    "{h}Subdomain {part} is left floating.{c}\n\
                    Its corner nodes and pinned degrees of freedom remove only {removed} of its \
                    6 rigid-body modes, so its stiffness is singular. A subdomain needs three \
                    non-collinear corner nodes (nodes shared by three or more subdomains) or \
                    enough pinned degrees of freedom, so the partition has to bring more \
                    subdomains together at a node."
                )
            }
            Self::SingularSubdomain(part) => {
                let (h, c) = (style.headline, style.frame);
                format!(
                    "{h}The stiffness of subdomain {part} is singular.{c}\n\
                    Its corners and pinned degrees of freedom hold the subdomain as a whole, but \
                    some part of it is still free to move: a piece attached to the rest through \
                    only a node or an edge, or cut off from it altogether. Change the partition \
                    so that every part of a subdomain is attached through faces. Otherwise the \
                    model itself has a mechanism or a collapsed element."
                )
            }
            Self::SingularCoarseProblem => {
                let (h, c) = (style.headline, style.frame);
                format!(
                    "{h}The problem coupling the corners is singular.{c}\n\
                    Pin enough degrees of freedom to remove every rigid-body mode of the \
                    whole block."
                )
            }
        }
    }
}

styled_error!(SolveError);

impl From<FiniteElementError> for SolveError {
    fn from(error: FiniteElementError) -> Self {
        Self::Element(error)
    }
}

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
#[allow(clippy::type_complexity)]
pub(crate) fn solve<C, F, const G: usize, const M: usize, const N: usize, const P: usize>(
    block: &Block<C, F, G, M, N, P>,
    nodal_coordinates: &NodalCoordinates<3>,
    partition: &Partition,
    boundary_conditions: &BoundaryConditions,
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
        Preconditioner::Dirichlet,
        Krylov::default().rel_tol,
    )
}

pub(crate) fn solve_with<C, F, const G: usize, const M: usize, const N: usize, const P: usize>(
    block: &Block<C, F, G, M, N, P>,
    nodal_coordinates: &NodalCoordinates<3>,
    partition: &Partition,
    boundary_conditions: &BoundaryConditions,
    preconditioner: Preconditioner,
    rel_tol: Scalar,
) -> Result<Vector, SolveError>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, M, N, P>,
{
    let (local_stiffnesses, local_forces): (Vec<SquareMatrix>, Vec<Vector>) = partition
        .parts_nodes()
        .iter()
        .map(|nodes| local_stiffness_and_force(block, nodal_coordinates, nodes))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();
    solve_local_systems(
        partition,
        boundary_conditions,
        local_stiffnesses,
        local_forces,
        &positions(nodal_coordinates),
        preconditioner,
        rel_tol,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_local_systems<const D: usize>(
    partition: &Partition,
    boundary_conditions: &BoundaryConditions,
    local_stiffnesses: Vec<SquareMatrix>,
    local_forces: Vec<Vector>,
    positions: &[[f64; D]],
    preconditioner: Preconditioner,
    rel_tol: Scalar,
) -> Result<Vector, SolveError> {
    let corners = CornerSelection::from_partition(partition);
    let (interfaces, num_multipliers) = build_interfaces(partition, &corners, D);
    let (splits, corner_dofs) = build_splits(partition, &corners, boundary_conditions, D);
    let subdomain_nodes = partition.parts_nodes();
    let removable = D + D * (D - 1) / 2;
    subdomain_nodes
        .iter()
        .zip(&splits)
        .enumerate()
        .try_for_each(|(part, (nodes, split))| {
            if nodes.is_empty() {
                return Ok(());
            }
            let free: HashSet<usize> = split.dual().iter().copied().collect();
            let constrained: Vec<usize> = (0..D * nodes.len())
                .filter(|dof| !free.contains(dof))
                .collect();
            let local: Vec<[f64; D]> = nodes.iter().map(|&node| positions[node]).collect();
            let removed = removed_modes(&local, &constrained);
            if removed < removable {
                Err(SolveError::FloatingSubdomain { part, removed })
            } else {
                Ok(())
            }
        })?;
    let indices: Vec<usize> = (0..subdomain_nodes.len()).collect();
    let condensed = parallel_map(&indices, THREADS, |&index| {
        Condensed::try_condense(
            &local_stiffnesses[index],
            &local_forces[index],
            splits[index].primal(),
            splits[index].dual(),
        )
    })
    .into_iter()
    .enumerate()
    .map(|(part, condensed)| condensed.ok_or(SolveError::SingularSubdomain(part)))
    .collect::<Result<Vec<_>, _>>()?;
    let (schur, reduced_force) = CoarseSystem::assemble(&condensed, &splits, &corner_dofs);
    let coarse_problem = Coarse::try_from(schur).map_err(|_| SolveError::SingularCoarseProblem)?;
    let locals = parallel_map(&indices, THREADS, |&index| {
        let stiffness = &local_stiffnesses[index];
        let dual_dofs = splits[index].dual().to_vec();
        let dual_stiffness: SquareMatrix = dual_dofs
            .iter()
            .map(|&row| dual_dofs.iter().map(|&col| stiffness[row][col]).collect())
            .collect();
        let dual_factor = dual_stiffness
            .factorize_lu()
            .expect("K_dd is singular, but corners should make every subdomain non-singular");
        let dirichlet = DirichletLocal::build(stiffness, &dual_dofs, interfaces[index].dofs());
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
                nodes.len() * D,
                condensed.dual_map.clone(),
                split.primal().to_vec(),
                split.primal_global().to_vec(),
                dirichlet,
            )
        })
        .collect();
    let rhs = rhs_from_forces(&subdomains, &local_forces, num_multipliers)
        - coupling(
            &subdomains,
            &coarse_problem.solve(&reduced_force),
            num_multipliers,
        );
    let lambda = projected_pcg_with(&subdomains, &coarse_problem, &rhs, preconditioner, rel_tol)?;
    let ct_lambda = coupling_transpose(&subdomains, &lambda, coarse_problem.len());
    let corner_solution = coarse_problem.solve(&(reduced_force + ct_lambda));
    let recovered = primal_recovery(&subdomains, &local_forces, &corner_solution, &lambda);
    let mut global = Vector::zero(positions.len() * D);
    subdomain_nodes
        .iter()
        .zip(recovered.iter())
        .for_each(|(nodes, local_solution)| {
            nodes.iter().enumerate().for_each(|(local, &node)| {
                (0..D).for_each(|component| {
                    global[D * node + component] = local_solution[D * local + component]
                })
            })
        });
    Ok(global)
}
