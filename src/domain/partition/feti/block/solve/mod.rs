#[cfg(test)]
mod test;

use super::super::{
    THREADS,
    dual::{coupling, coupling_transpose, rhs_from_forces},
    dual_primal::{self, BoundaryConditions, coarse::Coarse},
    interface,
    parallel::parallel_map,
    pcg::{Preconditioner, primal_recovery, projected_pcg_with},
    subdomain::{DirichletLocal, Subdomain},
};
use super::{assemble, element_systems};
use crate::{
    constitutive::solid::hyperelastic::Hyperelastic,
    fem::{
        NodalCoordinates,
        block::{
            Block,
            element::{FiniteElementError, solid::hyperelastic::HyperelasticFiniteElement},
        },
    },
    math::{
        Scalar, SquareMatrix, Style, StyledError, Vector,
        optimize::{Krylov, KrylovError},
        styled_error,
    },
};
use std::collections::HashSet;

/// Errors a full FETI-DP solve can hit: either extracting a subdomain's
/// local stiffness/force from the real `Block` fails, or the dual PCG does.
pub enum SolveError {
    Element(FiniteElementError),
    Krylov(KrylovError),
    /// A subdomain whose corners and pinned degrees of freedom leave some of
    /// its rigid-body modes free, and how many they remove of the six.
    FloatingSubdomain {
        part: usize,
        removed: usize,
    },
    SingularSubdomain(usize),
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
#[allow(clippy::type_complexity)]
pub(crate) fn solve<C, F, const G: usize, const M: usize, const N: usize, const P: usize>(
    block: &Block<C, F, G, M, N, P>,
    nodal_coordinates: &NodalCoordinates<3>,
    partition: &crate::geometry::mesh::Partition,
    boundary_conditions: &BoundaryConditions,
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
}

/// `solve` with the dual PCG preconditioner chosen explicitly.
pub(crate) fn solve_with<C, F, const G: usize, const M: usize, const N: usize, const P: usize>(
    block: &Block<C, F, G, M, N, P>,
    nodal_coordinates: &NodalCoordinates<3>,
    partition: &crate::geometry::mesh::Partition,
    boundary_conditions: &BoundaryConditions,
    dimension: usize,
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
        .map(|nodes| assemble::local_stiffness_and_force(block, nodal_coordinates, nodes))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();
    solve_local_systems(
        partition,
        boundary_conditions,
        local_stiffnesses,
        local_forces,
        &element_systems::positions(nodal_coordinates),
        dimension,
        preconditioner,
        rel_tol,
    )
}

/// The FETI-DP solve once each subdomain's local stiffness and force are in
/// hand, however they were assembled.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_local_systems(
    partition: &crate::geometry::mesh::Partition,
    boundary_conditions: &BoundaryConditions,
    local_stiffnesses: Vec<SquareMatrix>,
    local_forces: Vec<Vector>,
    positions: &[[f64; 3]],
    dimension: usize,
    preconditioner: Preconditioner,
    rel_tol: Scalar,
) -> Result<Vector, SolveError> {
    let corners = dual_primal::CornerSelection::from_partition(partition);
    let (interfaces, num_multipliers) = interface::build_interfaces(partition, &corners, dimension);
    let (splits, corner_dofs) =
        dual_primal::build_splits(partition, &corners, boundary_conditions, dimension);
    let subdomain_nodes = partition.parts_nodes();
    subdomain_nodes
        .iter()
        .zip(&splits)
        .enumerate()
        .try_for_each(|(part, (nodes, split))| {
            if nodes.is_empty() {
                return Ok(());
            }
            let free: HashSet<usize> = split.dual().iter().copied().collect();
            let constrained: Vec<usize> = (0..dimension * nodes.len())
                .filter(|dof| !free.contains(dof))
                .collect();
            let local: Vec<[f64; 3]> = nodes.iter().map(|&node| positions[node]).collect();
            let removed = dual_primal::rigid::removed_modes(&local, &constrained);
            if removed < 6 {
                Err(SolveError::FloatingSubdomain { part, removed })
            } else {
                Ok(())
            }
        })?;
    let indices: Vec<usize> = (0..subdomain_nodes.len()).collect();
    let condensed = parallel_map(&indices, THREADS, |&index| {
        dual_primal::condense::Condensed::try_condense(
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
    let (schur, reduced_force) =
        dual_primal::coarse::CoarseSystem::assemble(&condensed, &splits, &corner_dofs);
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
                nodes.len() * dimension,
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
    let mut global = Vector::zero(positions.len() * dimension);
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
    Ok(global)
}
