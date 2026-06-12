#[cfg(test)]
mod test;

pub mod element;
pub mod solid;
pub mod surface;
pub mod thermal;

use crate::{
    constitutive::Constitutive,
    fem::{
        FiniteElements, NodalReferenceCoordinates,
        block::element::{
            ElementNodalReferenceCoordinates, FiniteElement,
            planar::PlanarElementNodalReferenceCoordinates,
        },
    },
    math::{
        Banded, InverseSets, Scalar, SetsOld as Sets, Tensor, TensorRank1List, TensorRank1Vec,
        disjoint_set_union, optimize::EqualityConstraint,
    },
};
use std::{
    any::type_name,
    array::from_fn,
    fmt::{self, Debug, Formatter},
    iter::repeat_n,
};

pub type Connectivity<const N: usize> = Vec<[usize; N]>;
pub type Graph<const N: usize> = Sets<Vec<[usize; N]>, [usize; N], usize, Vec<usize>, usize>;

pub struct Block<C, F, const G: usize, const M: usize, const N: usize, const P: usize> {
    constitutive_model: C,
    connectivity: Graph<N>,
    elements: Vec<F>,
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> Block<C, F, G, M, N, P>
where
    F: FiniteElement<G, M, N, P>,
{
    fn constitutive_model(&self) -> &C {
        &self.constitutive_model
    }
    fn connectivity(&self) -> &Connectivity<N> {
        self.connectivity.members()
    }
    fn elements(&self) -> &[F] {
        &self.elements
    }
    fn element_coordinates<const D: usize, const I: usize>(
        coordinates: &TensorRank1Vec<D, I>,
        nodes: &[usize; N],
    ) -> TensorRank1List<D, I, N> {
        nodes
            .iter()
            .map(|&node| coordinates[node].clone())
            .collect()
    }
    pub fn volume(&self) -> Scalar {
        self.elements().iter().map(|element| element.volume()).sum()
    }
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> Debug
    for Block<C, F, G, M, N, P>
where
    F: FiniteElement<G, M, N, P>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Block {{ constitutive model: {}, {} elements }}",
            type_name::<C>()
                .rsplit("::")
                .next()
                .unwrap()
                .split("<")
                .next()
                .unwrap(),
            self.elements().len()
        )
    }
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> FiniteElements
    for Block<C, F, G, M, N, P>
where
    F: FiniteElement<G, M, N, P>,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        add_node_neighbors(self.connectivity(), neighbors)
    }
}

pub trait FiniteElementBlock<C, F, const G: usize, const N: usize>
where
    Self: for<'a> From<(C, Connectivity<N>, &'a NodalReferenceCoordinates<3>)>,
{
    fn isolate(
        self,
        elements: &[usize],
        coordinates: &NodalReferenceCoordinates<3>,
    ) -> Vec<(Self, NodalReferenceCoordinates<3>, [Vec<usize>; 3])>;
    fn reset(&mut self);
}

impl<C, F, const G: usize, const N: usize, const P: usize>
    From<(C, Connectivity<N>, &NodalReferenceCoordinates<3>)> for Block<C, F, G, 3, N, P>
where
    F: FiniteElement<G, 3, N, P> + From<ElementNodalReferenceCoordinates<N>>,
{
    fn from(
        (constitutive_model, connectivity, coordinates): (
            C,
            Connectivity<N>,
            &NodalReferenceCoordinates<3>,
        ),
    ) -> Self {
        let elements = connectivity
            .iter()
            .map(|nodes| Self::element_coordinates(coordinates, nodes).into())
            .collect();
        let connectivity = connectivity.into();
        Self {
            constitutive_model,
            connectivity,
            elements,
        }
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize>
    From<(C, Connectivity<N>, &NodalReferenceCoordinates<2>)> for Block<C, F, G, 2, N, P>
where
    F: FiniteElement<G, 2, N, P> + From<PlanarElementNodalReferenceCoordinates<N>>,
{
    fn from(
        (constitutive_model, connectivity, coordinates): (
            C,
            Connectivity<N>,
            &NodalReferenceCoordinates<2>,
        ),
    ) -> Self {
        let elements = connectivity
            .iter()
            .map(|nodes| Self::element_coordinates(coordinates, nodes).into())
            .collect();
        let connectivity = connectivity.into();
        Self {
            constitutive_model,
            connectivity,
            elements,
        }
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize> FiniteElementBlock<C, F, G, N>
    for Block<C, F, G, 3, N, P>
where
    C: Constitutive,
    F: Default + FiniteElement<G, 3, N, P> + From<ElementNodalReferenceCoordinates<N>>,
{
    fn isolate(
        self,
        isolated_elements: &[usize],
        coordinates: &NodalReferenceCoordinates<3>,
    ) -> Vec<(Self, NodalReferenceCoordinates<3>, [Vec<usize>; 3])> {
        let (graph, map) = self.connectivity.inverse();
        let (_, node_elements) = graph.into();
        let (_, element_nodes) = self.connectivity.into();
        let isolated_element_nodes: Connectivity<N> = isolated_elements
            .iter()
            .map(|&isolated_element| element_nodes[isolated_element])
            .collect();
        disjoint_set_union(&isolated_element_nodes, coordinates.len())
            .into_iter()
            .map(|isolated_nodes| {
                let mut block_elements = isolated_nodes
                    .iter()
                    .flat_map(|&node| node_elements[map[node]].iter().copied())
                    .collect::<Vec<_>>();
                block_elements.sort_unstable();
                block_elements.dedup();
                let mut global_nodes = block_elements
                    .iter()
                    .flat_map(|&element| element_nodes[element])
                    .collect::<Vec<_>>();
                global_nodes.sort_unstable();
                global_nodes.dedup();
                let constitutive_model = self.constitutive_model.clone();
                let mut node_num = 0;
                let mut local_nodes = vec![0; global_nodes.iter().max().unwrap() + 1];
                let block_coordinates = global_nodes
                    .iter()
                    .map(|&node| {
                        local_nodes[node] = node_num;
                        node_num += 1;
                        coordinates[node].clone()
                    })
                    .collect();
                let connectivity = block_elements
                    .iter()
                    .map(|&element| from_fn(|node| local_nodes[element_nodes[element][node]]))
                    .collect::<Vec<_>>()
                    .into();
                let elements = block_elements
                    .into_iter()
                    .map(|element| self.elements[element].clone())
                    .collect();
                let mut global_boundary_nodes = global_nodes.clone();
                global_boundary_nodes.retain(|node| isolated_nodes.binary_search(node).is_err());
                let boundary_nodes = global_boundary_nodes
                    .into_iter()
                    .map(|node| local_nodes[node])
                    .collect();
                (
                    Self {
                        constitutive_model,
                        connectivity,
                        elements,
                    },
                    block_coordinates,
                    [boundary_nodes, local_nodes, global_nodes],
                )
            })
            .collect()
    }
    fn reset(&mut self) {
        self.elements
            .iter_mut()
            .for_each(|element| *element = F::default())
    }
}

pub(crate) fn add_node_neighbors<const N: usize>(
    connectivity: &Connectivity<N>,
    neighbors: &mut [Vec<usize>],
) {
    connectivity.iter().for_each(|nodes| {
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

pub(crate) fn band_from_neighbors(
    neighbors: &[Vec<usize>],
    equality_constraint: &EqualityConstraint,
    dimension: usize,
) -> Banded {
    let number_of_nodes = neighbors.len();
    let structure: Vec<Vec<bool>> = neighbors
        .iter()
        .map(|nodes| (0..number_of_nodes).map(|b| nodes.contains(&b)).collect())
        .collect();
    let structure_nd: Vec<Vec<bool>> = structure
        .iter()
        .flat_map(|row| {
            repeat_n(
                row.iter()
                    .flat_map(|entry| repeat_n(*entry, dimension))
                    .collect(),
                dimension,
            )
        })
        .collect();
    match equality_constraint {
        EqualityConstraint::Fixed(indices) => {
            let mut keep = vec![true; structure_nd.len()];
            indices.iter().for_each(|&index| keep[index] = false);
            let banded = structure_nd
                .into_iter()
                .zip(keep.iter())
                .filter(|(_, keep)| **keep)
                .map(|(structure_nd_a, _)| {
                    structure_nd_a
                        .into_iter()
                        .zip(keep.iter())
                        .filter(|(_, keep)| **keep)
                        .map(|(structure_nd_ab, _)| structure_nd_ab)
                        .collect::<Vec<bool>>()
                })
                .collect::<Vec<Vec<bool>>>();
            Banded::from(banded)
        }
        EqualityConstraint::Linear(matrix, _) => {
            let num_coords = dimension * number_of_nodes;
            assert_eq!(matrix.width(), num_coords);
            let num_dof = matrix.len() + matrix.width();
            let mut banded = vec![vec![false; num_dof]; num_dof];
            structure_nd
                .iter()
                .zip(banded.iter_mut())
                .for_each(|(structure_nd_i, banded_i)| {
                    structure_nd_i
                        .iter()
                        .zip(banded_i.iter_mut())
                        .for_each(|(structure_nd_ij, banded_ij)| *banded_ij = *structure_nd_ij)
                });
            let mut index = num_coords;
            matrix.iter().for_each(|matrix_i| {
                matrix_i.iter().enumerate().for_each(|(j, matrix_ij)| {
                    if matrix_ij != &0.0 {
                        banded[index][j] = true;
                        banded[j][index] = true;
                        index += 1;
                    }
                })
            });
            Banded::from(banded)
        }
        EqualityConstraint::None => Banded::from(structure_nd),
    }
}
