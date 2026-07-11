#[cfg(test)]
mod test;

pub mod element;
pub mod solid;
pub mod surface;
pub mod thermal;

use crate::{
    fem::{
        Elements, NodalReferenceCoordinates,
        block::element::{
            ElementNodalReferenceCoordinates, FiniteElement,
            planar::PlanarElementNodalReferenceCoordinates,
        },
    },
    geometry::mesh::PrimitiveConnectivity,
    math::{Banded, Scalar, Tensor, TensorRank1List, TensorRank1Vec, optimize::EqualityConstraint},
};
use std::{
    any::type_name,
    fmt::{self, Debug, Formatter},
};

pub struct Block<C, F, const G: usize, const M: usize, const N: usize, const P: usize> {
    constitutive_model: C,
    connectivity: PrimitiveConnectivity<M, N>,
    elements: Vec<F>,
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> Block<C, F, G, M, N, P>
where
    F: FiniteElement<G, M, N, P>,
{
    fn constitutive_model(&self) -> &C {
        &self.constitutive_model
    }
    fn connectivity(&self) -> &PrimitiveConnectivity<M, N> {
        &self.connectivity
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

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> Elements
    for Block<C, F, G, M, N, P>
where
    F: FiniteElement<G, M, N, P>,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        add_node_neighbors(self.connectivity(), neighbors)
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize>
    From<(
        C,
        PrimitiveConnectivity<3, N>,
        &NodalReferenceCoordinates<3>,
    )> for Block<C, F, G, 3, N, P>
where
    F: FiniteElement<G, 3, N, P> + From<ElementNodalReferenceCoordinates<N>>,
{
    fn from(
        (constitutive_model, connectivity, coordinates): (
            C,
            PrimitiveConnectivity<3, N>,
            &NodalReferenceCoordinates<3>,
        ),
    ) -> Self {
        let elements = connectivity
            .iter()
            .map(|nodes| Self::element_coordinates(coordinates, nodes).into())
            .collect();
        Self {
            constitutive_model,
            connectivity,
            elements,
        }
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize>
    From<(C, Vec<[usize; N]>, &NodalReferenceCoordinates<3>)> for Block<C, F, G, 3, N, P>
where
    F: FiniteElement<G, 3, N, P> + From<ElementNodalReferenceCoordinates<N>>,
{
    fn from(
        (constitutive_model, connectivity, coordinates): (
            C,
            Vec<[usize; N]>,
            &NodalReferenceCoordinates<3>,
        ),
    ) -> Self {
        Self::from((
            constitutive_model,
            PrimitiveConnectivity::from(connectivity),
            coordinates,
        ))
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize>
    From<(
        C,
        PrimitiveConnectivity<2, N>,
        &NodalReferenceCoordinates<2>,
    )> for Block<C, F, G, 2, N, P>
where
    F: FiniteElement<G, 2, N, P> + From<PlanarElementNodalReferenceCoordinates<N>>,
{
    fn from(
        (constitutive_model, connectivity, coordinates): (
            C,
            PrimitiveConnectivity<2, N>,
            &NodalReferenceCoordinates<2>,
        ),
    ) -> Self {
        let elements = connectivity
            .iter()
            .map(|nodes| Self::element_coordinates(coordinates, nodes).into())
            .collect();
        Self {
            constitutive_model,
            connectivity,
            elements,
        }
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize>
    From<(C, Vec<[usize; N]>, &NodalReferenceCoordinates<2>)> for Block<C, F, G, 2, N, P>
where
    F: FiniteElement<G, 2, N, P> + From<PlanarElementNodalReferenceCoordinates<N>>,
{
    fn from(
        (constitutive_model, connectivity, coordinates): (
            C,
            Vec<[usize; N]>,
            &NodalReferenceCoordinates<2>,
        ),
    ) -> Self {
        Self::from((
            constitutive_model,
            PrimitiveConnectivity::from(connectivity),
            coordinates,
        ))
    }
}

pub(crate) fn add_node_neighbors<const M: usize, const N: usize>(
    connectivity: &PrimitiveConnectivity<M, N>,
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
            Banded::from_pattern(next, pattern)
        }
        EqualityConstraint::Linear(matrix, _) => {
            assert_eq!(matrix.width(), num_coords);
            let num_dof = matrix.len() + matrix.width();
            let mut index = num_coords;
            matrix.iter().for_each(|matrix_i| {
                matrix_i.iter().enumerate().for_each(|(j, matrix_ij)| {
                    if matrix_ij != &0.0 {
                        pattern.push((index, j));
                        pattern.push((j, index));
                        index += 1;
                    }
                })
            });
            Banded::from_pattern(num_dof, pattern)
        }
        EqualityConstraint::None => Banded::from_pattern(num_coords, pattern),
    }
}
