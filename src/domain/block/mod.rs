#[cfg(test)]
pub(crate) mod test;

pub(crate) mod element;
pub(crate) mod solid;

use crate::math::{
    TensorRank1List, TensorRank1Vec, optimize::EqualityConstraint, sparse::SparseSolver,
};
use std::any::type_name;

#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) fn trimmed_type_name<T>() -> &'static str {
    type_name::<T>()
        .rsplit("::")
        .next()
        .unwrap()
        .split('<')
        .next()
        .unwrap()
}

#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) fn element_coordinates<const D: usize, const N: usize, I, U>(
    coordinates: &TensorRank1Vec<D, I, U>,
    nodes: &[usize; N],
) -> TensorRank1List<D, I, N, U> {
    nodes
        .iter()
        .map(|&node| coordinates[node].clone())
        .collect()
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

pub(crate) fn solver_from_neighbors(
    neighbors: &[Vec<usize>],
    equality_constraints: &[EqualityConstraint],
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
    let merged;
    let constraints = if equality_constraints.len() > 1 {
        merged = EqualityConstraint::merged(equality_constraints.to_vec());
        merged.as_slice()
    } else {
        equality_constraints
    };
    match constraints {
        [] | [EqualityConstraint::None] => {
            SparseSolver::from_pattern(num_coords, pattern, symmetric)
        }
        [EqualityConstraint::Fixed(_)] | [EqualityConstraint::Prescribed(_)] => {
            let Some(indices) = EqualityConstraint::held_indices(constraints) else {
                unreachable!("A fixed or prescribed constraint holds indices.")
            };
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
        parts => {
            let mut rows = Vec::<Vec<usize>>::new();
            parts.iter().for_each(|part| match part {
                EqualityConstraint::Multipliers(entries) => {
                    rows.extend(entries.iter().map(|&(index, _)| vec![index]))
                }
                EqualityConstraint::Linear(matrix, _) => {
                    assert_eq!(matrix.width(), num_coords);
                    rows.extend((0..matrix.len()).map(|row| {
                        (0..num_coords)
                            .filter(|&column| matrix[row][column] != 0.0)
                            .collect()
                    }))
                }
                _ => {}
            });
            rows.iter().enumerate().for_each(|(row, columns)| {
                columns.iter().for_each(|&column| {
                    pattern.push((num_coords + row, column));
                    pattern.push((column, num_coords + row));
                })
            });
            SparseSolver::from_pattern(num_coords + rows.len(), pattern, symmetric)
        }
    }
}
