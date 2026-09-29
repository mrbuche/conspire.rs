#[cfg(test)]
mod test;

use super::super::{
    Formulation, GMRES, THREADS,
    dual::{coupling, coupling_transpose, rhs_from_forces},
    dual_primal::{
        BoundaryConditions, CornerSelection, Row, build_splits,
        coarse::{Coarse, CoarseSystem, CornerConstraint},
        condense::Condensed,
        rigid::{kernel, kernel_pins, removed_modes},
        rigid_projector::{RigidProjector, add_rigid_motion, rigid_rhs},
        select::select_corners,
    },
    interface::build_interfaces,
    parallel::parallel_map,
    pcg::{Preconditioner, primal_recovery, projected_pcg_with, rigid_projected_pcg},
    subdomain::{DirichletLocal, Subdomain},
};
use crate::{
    domain::ElementModelError,
    geometry::mesh::Partition,
    math::{
        Scalar, SquareMatrix, Style, StyledError, Tensor, Vector,
        optimize::{KrylovError, KrylovMethod},
        styled_error,
    },
};
use std::collections::{HashMap, HashSet};

/// Possible errors encountered when solving with FETI.
pub enum SolveError {
    /// Downstream error from an element.
    Element(ElementModelError),
    /// The partition does not describe the model it is meant to decompose.
    Partition(String),
    /// Downstream error from the dual PCG.
    Krylov(KrylovError),
    /// A subdomain left with some rigid-body modes free.
    FloatingSubdomain { part: usize, removed: usize },
    /// A subdomain with some part of it still free to move.
    SingularSubdomain(usize),
    /// The interior of a subdomain, away from the interface, is singular.
    SingularInterior(usize),
    /// The assembled corner problem is singular.
    SingularCoarseProblem,
}

impl StyledError for SolveError {
    fn message(&self, style: &Style) -> String {
        match self {
            Self::Element(error) => error.message(style),
            Self::Partition(reason) => {
                let (h, c) = (style.headline, style.frame);
                format!("{h}The partition does not fit the model.{c}\n{reason}")
            }
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
            Self::SingularInterior(part) => {
                let (h, c) = (style.headline, style.frame);
                format!(
                    "{h}The interior of subdomain {part} is singular.{c}\n\
                    The degrees of freedom of the subdomain away from the interface form a \
                    singular block, though the subdomain as a whole does not, which the \
                    Dirichlet preconditioner cannot handle. The tangent is likely not \
                    positive definite there, as under severe compression, or the partition \
                    leaves part of the interior loosely attached."
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

impl From<ElementModelError> for SolveError {
    fn from(error: ElementModelError) -> Self {
        Self::Element(error)
    }
}

impl From<KrylovError> for SolveError {
    fn from(error: KrylovError) -> Self {
        Self::Krylov(error)
    }
}

const RELATIVE_PIVOT: Scalar = 1e-10;

enum LocalError {
    Singular,
    Interior,
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
    method: KrylovMethod,
    formulation: Formulation,
) -> Result<(Vector, Vector), SolveError> {
    let rows = boundary_conditions.rows();
    if let Some(&(node, component, _)) = rows
        .iter()
        .flat_map(|row| row.entries.iter())
        .find(|&&(node, component, _)| node >= positions.len() || component >= D)
    {
        return Err(SolveError::Partition(format!(
            "constraint on node {node}, component {component}, which the model does not have"
        )));
    }
    let corners = match formulation {
        Formulation::Classical => CornerSelection::new(Vec::new()),
        Formulation::DualPrimal => select_corners(partition, positions, boundary_conditions, &rows),
    };
    let (mut interfaces, num_interface_multipliers) = build_interfaces(partition, &corners, D);
    let mut home = HashMap::<usize, (usize, usize)>::new();
    partition
        .parts_nodes()
        .iter()
        .enumerate()
        .for_each(|(part, nodes)| {
            nodes.iter().enumerate().for_each(|(local, &node)| {
                home.entry(node).or_insert((part, local));
            })
        });
    let (dual_rows, primal_rows): (Vec<&Row>, Vec<&Row>) = rows.iter().partition(|row| {
        row.single()
            .is_some_and(|(node, _, _)| !corners.contains(node))
    });
    if formulation == Formulation::Classical && !primal_rows.is_empty() {
        return Err(SolveError::Partition(
            "classical FETI takes constraints of a single DOF only".to_string(),
        ));
    }
    let method = match method {
        KrylovMethod::ConjugateGradients if !dual_rows.is_empty() => GMRES,
        method => method,
    };
    let mut num_multipliers = num_interface_multipliers;
    let mut prescribed = Vector::zero(dual_rows.len());
    dual_rows.iter().enumerate().for_each(|(k, row)| {
        let (node, component, coefficient) = row.single().expect("a row of one entry");
        let (part, local) = home
            .get(&node)
            .copied()
            .expect("a constrained node is in no subdomain");
        interfaces[part].push(num_multipliers, D * local + component, coefficient);
        prescribed[k] = row.value;
        num_multipliers += 1;
    });
    let (splits, corner_dofs) = build_splits(partition, &corners, boundary_conditions, D);
    let subdomain_nodes = partition.parts_nodes();
    let removable = D + D * (D - 1) / 2;
    subdomain_nodes
        .iter()
        .zip(&splits)
        .enumerate()
        .try_for_each(|(part, (nodes, split))| {
            if nodes.is_empty() || formulation == Formulation::Classical {
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
        if formulation == Formulation::Classical {
            return Some(Condensed::without_corners(splits[index].dual().len()));
        }
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
    let constraints = primal_rows
        .iter()
        .map(|row| CornerConstraint {
            entries: row
                .entries
                .iter()
                .map(|&(node, component, coefficient)| {
                    (
                        corner_dofs
                            .global_index(node, component)
                            .expect("a constrained corner has a DOF"),
                        coefficient,
                    )
                })
                .collect(),
            value: row.value,
        })
        .collect::<Vec<_>>();
    let constraint_values: Vector = constraints.iter().map(|row| row.value).collect();
    let coarse_problem =
        Coarse::try_new(schur, &constraints).map_err(|_| SolveError::SingularCoarseProblem)?;
    let locals = parallel_map(&indices, THREADS, |&index| {
        let stiffness = &local_stiffnesses[index];
        let dual_dofs = splits[index].dual().to_vec();
        let dual_stiffness: SquareMatrix = dual_dofs
            .iter()
            .map(|&row| dual_dofs.iter().map(|&col| stiffness[row][col]).collect())
            .collect();
        let (dual_factor, floating) = if formulation == Formulation::Classical {
            let free: HashSet<usize> = dual_dofs.iter().copied().collect();
            let constrained: Vec<usize> = (0..D * subdomain_nodes[index].len())
                .filter(|dof| !free.contains(dof))
                .collect();
            let local: Vec<[f64; D]> = subdomain_nodes[index]
                .iter()
                .map(|&node| positions[node])
                .collect();
            let kernel = kernel(&local, &constrained);
            let pins = kernel_pins(&kernel, &dual_dofs);
            let keep: Vec<usize> = (0..dual_dofs.len())
                .filter(|position| pins.binary_search(position).is_err())
                .collect();
            let reduced: SquareMatrix = keep
                .iter()
                .map(|&row| keep.iter().map(|&col| dual_stiffness[row][col]).collect())
                .collect();
            let factor = reduced
                .factorize_lu()
                .ok()
                .filter(|factor| factor.near_zero_pivots(RELATIVE_PIVOT) == 0)
                .ok_or(LocalError::Singular)?;
            if kernel.is_empty() {
                (factor, None)
            } else {
                (factor, Some((kernel, keep)))
            }
        } else {
            let factor = dual_stiffness
                .factorize_lu()
                .expect("K_dd is singular, but corners should make every subdomain non-singular");
            (factor, None)
        };
        DirichletLocal::try_build(stiffness, &dual_dofs, interfaces[index].dofs())
            .map(|dirichlet| (dual_dofs, dual_stiffness, dual_factor, dirichlet, floating))
            .ok_or(LocalError::Interior)
    })
    .into_iter()
    .enumerate()
    .map(|(part, local)| {
        local.map_err(|error| match error {
            LocalError::Singular => SolveError::SingularSubdomain(part),
            LocalError::Interior => SolveError::SingularInterior(part),
        })
    })
    .collect::<Result<Vec<_>, _>>()?;
    let subdomains: Vec<Subdomain<()>> = interfaces
        .into_iter()
        .zip(locals)
        .zip(splits.iter())
        .zip(condensed.iter())
        .zip(subdomain_nodes.iter())
        .map(|((((interface, local), split), condensed), nodes)| {
            let (dual_dofs, dual_stiffness, dual_factor, dirichlet, floating) = local;
            let subdomain = Subdomain::new(
                (),
                interface,
                dual_stiffness,
                dual_factor,
                dual_dofs,
                nodes.len() * D,
                condensed.dual_map.clone(),
                condensed.primal_map.clone(),
                split.primal().to_vec(),
                split.primal_global().to_vec(),
                dirichlet,
            );
            match floating {
                Some((kernel, keep)) => subdomain.with_kernel(kernel, keep),
                None => subdomain,
            }
        })
        .collect();
    let mut rhs = rhs_from_forces(&subdomains, &local_forces, num_multipliers)
        - coupling(
            &subdomains,
            &coarse_problem
                .solve_constrained(&reduced_force, &constraint_values)
                .0,
            num_multipliers,
        );
    prescribed
        .iter()
        .enumerate()
        .for_each(|(k, value)| rhs[num_interface_multipliers + k] -= value);
    let (lambda, alpha) = match formulation {
        Formulation::DualPrimal => (
            projected_pcg_with(
                &subdomains,
                &coarse_problem,
                &rhs,
                preconditioner,
                rel_tol,
                method,
            )?,
            None,
        ),
        Formulation::Classical => {
            let projector =
                RigidProjector::try_new(&subdomains).ok_or(SolveError::SingularCoarseProblem)?;
            let (lambda, alpha) = rigid_projected_pcg(
                &subdomains,
                &projector,
                &rhs,
                &rigid_rhs(&subdomains, &local_forces),
                preconditioner,
                rel_tol,
                method,
            )?;
            (lambda, Some(alpha))
        }
    };
    let ct_lambda = coupling_transpose(&subdomains, &lambda, coarse_problem.len());
    let (corner_solution, corner_multipliers) =
        coarse_problem.solve_constrained(&(reduced_force + ct_lambda), &constraint_values);
    let mut recovered = primal_recovery(&subdomains, &local_forces, &corner_solution, &lambda);
    if let Some(alpha) = alpha {
        add_rigid_motion(&subdomains, &alpha, &mut recovered);
    }
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
    let mut multipliers = Vector::zero(boundary_conditions.num_constraints());
    primal_rows
        .iter()
        .zip(corner_multipliers.iter())
        .for_each(|(row, &multiplier)| multipliers[row.index] = multiplier);
    dual_rows
        .iter()
        .enumerate()
        .for_each(|(k, row)| multipliers[row.index] = lambda[num_interface_multipliers + k]);
    Ok((global, multipliers))
}
