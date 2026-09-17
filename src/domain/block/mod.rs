pub(crate) mod element;
pub(crate) mod solid;
#[cfg(test)]
pub(crate) mod test;

use crate::math::{Tensor, optimize::EqualityConstraint, sparse::SparseSolver};
use std::any::type_name;

pub(crate) fn trimmed_type_name<T>() -> &'static str {
    type_name::<T>()
        .rsplit("::")
        .next()
        .unwrap()
        .split('<')
        .next()
        .unwrap()
}

pub(crate) fn add_node_neighbors<'a>(
    elements_nodes: impl Iterator<Item = &'a [usize]>,
    neighbors: &mut [Vec<usize>],
) {
    elements_nodes.for_each(|nodes| {
        nodes.iter().for_each(|&node_a| {
            nodes
                .iter()
                .for_each(|&node_b| neighbors[node_a].push(node_b))
        })
    })
}

pub(crate) fn finalize_node_neighbors(neighbors: &mut [Vec<usize>]) {
    neighbors.iter_mut().for_each(|nodes| {
        nodes.sort_unstable();
        nodes.dedup();
    })
}

/// The sparse solver for the positions a mesh makes nonzero.
pub(crate) fn solver_from_neighbors(
    neighbors: &[Vec<usize>],
    equality_constraint: &EqualityConstraint,
    dimension: usize,
    symmetric: bool,
) -> SparseSolver {
    let number_of_nodes = neighbors.len();
    let num_coords = dimension * number_of_nodes;
    let mut pattern: Vec<(usize, usize)> = neighbors
        .iter()
        .enumerate()
        .flat_map(|(a, nodes)| {
            nodes.iter().flat_map(move |&b| {
                (0..dimension).flat_map(move |i| {
                    (0..dimension).map(move |j| (dimension * a + i, dimension * b + j))
                })
            })
        })
        .collect();
    match equality_constraint {
        EqualityConstraint::Fixed(indices) => {
            let mut keep = vec![true; num_coords];
            indices.iter().for_each(|&index| keep[index] = false);
            let mut remap = vec![0; num_coords];
            let mut next = 0;
            (0..num_coords).for_each(|i| {
                if keep[i] {
                    remap[i] = next;
                    next += 1;
                }
            });
            pattern.retain(|&(i, j)| keep[i] && keep[j]);
            let pattern = pattern
                .into_iter()
                .map(|(i, j)| (remap[i], remap[j]))
                .collect();
            SparseSolver::from_pattern(next, pattern, symmetric)
        }
        EqualityConstraint::Linear(matrix, _) => {
            assert_eq!(matrix.width(), num_coords);
            let num_dof = matrix.len() + matrix.width();
            matrix.iter().enumerate().for_each(|(row, matrix_i)| {
                let index = num_coords + row;
                matrix_i.iter().enumerate().for_each(|(j, matrix_ij)| {
                    if matrix_ij != &0.0 {
                        pattern.push((index, j));
                        pattern.push((j, index));
                    }
                })
            });
            SparseSolver::from_pattern(num_dof, pattern, symmetric)
        }
        EqualityConstraint::None => SparseSolver::from_pattern(num_coords, pattern, symmetric),
    }
}
