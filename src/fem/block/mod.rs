#[cfg(test)]
mod test;

pub mod element;
pub mod solid;
pub mod thermal;

use super::*;
use crate::{
    defeat_message,
    fem::block::element::{FiniteElement, SurfaceFiniteElement},
    math::{
        Banded, TestError,
        optimize::{
            EqualityConstraint, FirstOrderOptimization, FirstOrderRootFinding, OptimizationError,
            SecondOrderOptimization, ZerothOrderRootFinding,
        },
    },
};
use std::{
    any::type_name,
    fmt::{self, Debug, Display, Formatter},
    iter::repeat_n,
};

pub type Connectivity<const N: usize> = Vec<[usize; N]>;

pub struct ElementBlock<C, F, const N: usize> {
    constitutive_model: C,
    connectivity: Connectivity<N>,
    coordinates: ReferenceNodalCoordinates,
    elements: Vec<F>,
}

impl<C, F, const N: usize> ElementBlock<C, F, N> {
    fn constitutive_model(&self) -> &C {
        &self.constitutive_model
    }
    fn connectivity(&self) -> &Connectivity<N> {
        &self.connectivity
    }
    fn coordinates(&self) -> &ReferenceNodalCoordinates {
        &self.coordinates
    }
    fn elements(&self) -> &[F] {
        &self.elements
    }
}

impl<C, F, const N: usize> Debug for ElementBlock<C, F, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let element = match N {
            3 => "LinearTriangle",
            4 => "LinearTetrahedron",
            8 => "LinearHexahedron",
            10 => "CompositeTetrahedron",
            _ => panic!(),
        };
        write!(
            f,
            "ElementBlock {{ constitutive_model: {}, elements: [{element}; {}] }}",
            type_name::<C>()
                .rsplit("::")
                .next()
                .unwrap()
                .split("<")
                .next()
                .unwrap(),
            self.connectivity.len()
        )
    }
}

pub trait FiniteElementBlock<C, F, const G: usize, const N: usize>
where
    F: FiniteElement<G, N>,
{
    fn new(
        constitutive_model: C,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: ReferenceNodalCoordinates,
    ) -> Self;
    fn reset(&mut self);
}

pub trait SurfaceFiniteElementBlock<C, F, const G: usize, const N: usize, const P: usize>
where
    F: SurfaceFiniteElement<G, N, P>,
{
    fn new(
        constitutive_model: C,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: ReferenceNodalCoordinates,
        thickness: Scalar,
    ) -> Self;
}

pub enum FiniteElementBlockError {
    Upstream(String, String),
}

impl From<FiniteElementBlockError> for String {
    fn from(error: FiniteElementBlockError) -> Self {
        match error {
            FiniteElementBlockError::Upstream(error, block) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In finite element block: {block}."
                )
            }
        }
    }
}

impl From<FiniteElementBlockError> for TestError {
    fn from(error: FiniteElementBlockError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl Debug for FiniteElementBlockError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Upstream(error, block) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In block: {block}."
                )
            }
        };
        write!(f, "\n{error}\n\x1b[0;2;31m{}\x1b[0m\n", defeat_message())
    }
}

impl Display for FiniteElementBlockError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Upstream(error, block) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In block: {block}."
                )
            }
        };
        write!(f, "{error}\x1b[0m")
    }
}

impl<C, F, const G: usize, const N: usize> FiniteElementBlock<C, F, G, N> for ElementBlock<C, F, N>
where
    F: FiniteElement<G, N>,
{
    fn new(
        constitutive_model: C,
        connectivity: Connectivity<N>,
        coordinates: ReferenceNodalCoordinates,
    ) -> Self {
        let elements = connectivity
            .iter()
            .map(|element_connectivity| {
                <F>::from(
                    element_connectivity
                        .iter()
                        .map(|&node| coordinates[node].clone())
                        .collect(),
                )
            })
            .collect();
        Self {
            constitutive_model,
            connectivity,
            coordinates,
            elements,
        }
    }
    fn reset(&mut self) {
        self.elements.iter_mut().for_each(|element| element.reset())
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize> SurfaceFiniteElementBlock<C, F, G, N, P>
    for ElementBlock<C, F, N>
where
    F: SurfaceFiniteElement<G, N, P>,
{
    fn new(
        constitutive_model: C,
        connectivity: Connectivity<N>,
        coordinates: ReferenceNodalCoordinates,
        thickness: Scalar,
    ) -> Self {
        let elements = connectivity
            .iter()
            .map(|element_connectivity| {
                <F>::new(
                    element_connectivity
                        .iter()
                        .map(|&node| coordinates[node].clone())
                        .collect(),
                    thickness,
                )
            })
            .collect();
        Self {
            constitutive_model,
            connectivity,
            coordinates,
            elements,
        }
    }
}

pub trait ZerothOrderRoot<C, E, const G: usize, const N: usize, X> {
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl ZerothOrderRootFinding<X>,
    ) -> Result<X, OptimizationError>;
}

pub trait FirstOrderRoot<C, E, const G: usize, const N: usize, F, J, X> {
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderRootFinding<F, J, X>,
    ) -> Result<X, OptimizationError>;
}

pub trait FirstOrderMinimize<C, E, const G: usize, const N: usize, X> {
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderOptimization<Scalar, X>,
    ) -> Result<X, OptimizationError>;
}

pub trait SecondOrderMinimize<C, E, const G: usize, const N: usize, J, H, X> {
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl SecondOrderOptimization<Scalar, J, H, X>,
    ) -> Result<X, OptimizationError>;
}

fn band<const N: usize>(
    connectivity: &Connectivity<N>,
    equality_constraint: &EqualityConstraint,
    number_of_nodes: usize,
    dimension: usize,
) -> Banded {
    match equality_constraint {
        EqualityConstraint::Fixed(indices) => {
            let neighbors: Vec<Vec<usize>> = invert(connectivity, number_of_nodes)
                .iter()
                .map(|elements| {
                    let mut nodes: Vec<usize> = elements
                        .iter()
                        .flat_map(|&element| connectivity[element])
                        .collect();
                    nodes.sort();
                    nodes.dedup();
                    nodes
                })
                .collect();
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
            let neighbors: Vec<Vec<usize>> = invert(connectivity, number_of_nodes)
                .iter()
                .map(|elements| {
                    let mut nodes: Vec<usize> = elements
                        .iter()
                        .flat_map(|&element| connectivity[element])
                        .collect();
                    nodes.sort();
                    nodes.dedup();
                    nodes
                })
                .collect();
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
        EqualityConstraint::None => {
            let neighbors: Vec<Vec<usize>> = invert(connectivity, number_of_nodes)
                .iter()
                .map(|elements| {
                    let mut nodes: Vec<usize> = elements
                        .iter()
                        .flat_map(|&element| connectivity[element])
                        .collect();
                    nodes.sort();
                    nodes.dedup();
                    nodes
                })
                .collect();
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
            Banded::from(structure_nd)
        }
    }
}

fn invert<const N: usize>(
    connectivity: &Connectivity<N>,
    number_of_nodes: usize,
) -> Vec<Vec<usize>> {
    let mut inverse_connectivity = vec![vec![]; number_of_nodes];
    connectivity
        .iter()
        .enumerate()
        .for_each(|(element, nodes)| {
            nodes
                .iter()
                .for_each(|&node| inverse_connectivity[node].push(element))
        });
    inverse_connectivity
}
