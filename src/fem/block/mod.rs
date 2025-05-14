#[cfg(test)]
mod test;

pub mod element;

use self::element::{
    ElasticFiniteElement, ElasticHyperviscousFiniteElement, FiniteElement, FiniteElementMethods,
    HyperelasticFiniteElement, HyperviscoelasticFiniteElement, SurfaceFiniteElement,
    ViscoelasticFiniteElement,
};
use super::*;
use crate::math::{Banded, optimize::{
    EqualityConstraint, FirstOrderRootFinding, NewtonRaphson, OptimizeError,
    SecondOrderOptimization,
}};
use std::{array::from_fn, iter::repeat_n};

pub struct ElementBlock<F, const N: usize> {
    connectivity: Connectivity<N>,
    elements: Vec<F>,
}

pub trait FiniteElementBlockMethods<C, F, const G: usize, const N: usize>
where
    F: FiniteElementMethods<C, G, N>,
{
    fn connectivity(&self) -> &Connectivity<N>;
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Vec<DeformationGradients<G>>;
    fn elements(&self) -> &[F];
    fn nodal_coordinates_element(
        &self,
        element_connectivity: &[usize; N],
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> NodalCoordinates<N>;
}

pub trait FiniteElementBlock<C, F, const G: usize, const N: usize, Y>
where
    C: Constitutive<Y>,
    F: FiniteElement<C, G, N, Y>,
    Y: Parameters,
{
    fn new(
        constitutive_model_parameters: Y,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: ReferenceNodalCoordinatesBlock,
    ) -> Self;
}

pub trait SurfaceFiniteElementBlock<C, F, const G: usize, const N: usize, const P: usize, Y>
where
    C: Constitutive<Y>,
    F: SurfaceFiniteElement<C, G, N, P, Y>,
    Y: Parameters,
{
    fn new(
        constitutive_model_parameters: Y,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: ReferenceNodalCoordinatesBlock,
        thickness: Scalar,
    ) -> Self;
}

impl<C, F, const G: usize, const N: usize> FiniteElementBlockMethods<C, F, G, N>
    for ElementBlock<F, N>
where
    F: FiniteElementMethods<C, G, N>,
{
    fn connectivity(&self) -> &Connectivity<N> {
        &self.connectivity
    }
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Vec<DeformationGradients<G>> {
        self.elements()
            .iter()
            .zip(self.connectivity().iter())
            .map(|(element, element_connectivity)| {
                element.deformation_gradients(
                    &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                )
            })
            .collect()
    }
    fn elements(&self) -> &[F] {
        &self.elements
    }
    fn nodal_coordinates_element(
        &self,
        element_connectivity: &[usize; N],
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> NodalCoordinates<N> {
        element_connectivity
            .iter()
            .map(|node| nodal_coordinates[*node].clone())
            .collect()
    }
}

impl<C, F, const G: usize, const N: usize, Y> FiniteElementBlock<C, F, G, N, Y>
    for ElementBlock<F, N>
where
    C: Constitutive<Y>,
    F: FiniteElement<C, G, N, Y>,
    Y: Parameters,
{
    fn new(
        constitutive_model_parameters: Y,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: ReferenceNodalCoordinatesBlock,
    ) -> Self {
        let elements = connectivity
            .iter()
            .map(|element_connectivity| {
                <F>::new(
                    constitutive_model_parameters,
                    element_connectivity
                        .iter()
                        .map(|&node| reference_nodal_coordinates[node].clone())
                        .collect(),
                )
            })
            .collect();
        Self {
            connectivity: connectivity,
            elements,
        }
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize, Y>
    SurfaceFiniteElementBlock<C, F, G, N, P, Y> for ElementBlock<F, N>
where
    C: Constitutive<Y>,
    F: SurfaceFiniteElement<C, G, N, P, Y>,
    Y: Parameters,
{
    fn new(
        constitutive_model_parameters: Y,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: ReferenceNodalCoordinatesBlock,
        thickness: Scalar,
    ) -> Self {
        let elements = connectivity
            .iter()
            .map(|element_connectivity| {
                <F>::new(
                    constitutive_model_parameters,
                    element_connectivity
                        .iter()
                        .map(|node| reference_nodal_coordinates[*node].clone())
                        .collect(),
                    &thickness,
                )
            })
            .collect();
        Self {
            connectivity,
            elements,
        }
    }
}

pub trait ElasticFiniteElementBlock<C, F, const G: usize, const N: usize>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, N>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<NodalForcesBlock, ConstitutiveError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<NodalStiffnessesBlock, ConstitutiveError>;
    fn root(
        &self,
        initial_coordinates: NodalCoordinatesBlock,
        root_finding: NewtonRaphson,
        equality_constraint: EqualityConstraint,
    ) -> Result<NodalCoordinatesBlock, OptimizeError>;
}

pub trait HyperelasticFiniteElementBlock<C, F, const G: usize, const N: usize>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, N>,
    Self: ElasticFiniteElementBlock<C, F, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<Scalar, ConstitutiveError>;
    fn minimize(
        &self,
        initial_coordinates: NodalCoordinatesBlock,
        optimization: NewtonRaphson,
        equality_constraint: EqualityConstraint,
    ) -> Result<NodalCoordinatesBlock, OptimizeError>;
}

pub trait ViscoelasticFiniteElementBlock<C, F, const G: usize, const N: usize>
where
    C: Viscoelastic,
    F: ViscoelasticFiniteElement<C, G, N>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        nodal_velocities: &NodalVelocitiesBlock,
    ) -> Result<NodalForcesBlock, ConstitutiveError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        nodal_velocities: &NodalVelocitiesBlock,
    ) -> Result<NodalStiffnessesBlock, ConstitutiveError>;
    fn nodal_velocities_element(
        &self,
        element_connectivity: &[usize; N],
        nodal_velocities: &NodalVelocitiesBlock,
    ) -> NodalVelocities<N>;
}

pub trait ElasticHyperviscousFiniteElementBlock<C, F, const G: usize, const N: usize>
where
    C: ElasticHyperviscous,
    F: ElasticHyperviscousFiniteElement<C, G, N>,
    Self: ViscoelasticFiniteElementBlock<C, F, G, N>,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        nodal_velocities: &NodalVelocitiesBlock,
    ) -> Result<Scalar, ConstitutiveError>;
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        nodal_velocities: &NodalVelocitiesBlock,
    ) -> Result<Scalar, ConstitutiveError>;
}

pub trait HyperviscoelasticFiniteElementBlock<C, F, const G: usize, const N: usize>
where
    C: Hyperviscoelastic,
    F: HyperviscoelasticFiniteElement<C, G, N>,
    Self: ElasticHyperviscousFiniteElementBlock<C, F, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<Scalar, ConstitutiveError>;
}

impl<C, F, const G: usize, const N: usize> ElasticFiniteElementBlock<C, F, G, N>
    for ElementBlock<F, N>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, N>,
    Self: FiniteElementBlockMethods<C, F, G, N>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<NodalForcesBlock, ConstitutiveError> {
        let mut nodal_forces = NodalForcesBlock::zero(nodal_coordinates.len());
        self.elements()
            .iter()
            .zip(self.connectivity().iter())
            .try_for_each(|(element, element_connectivity)| {
                element
                    .nodal_forces(
                        &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                    )?
                    .iter()
                    .zip(element_connectivity.iter())
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), ConstitutiveError>(())
            })?;
        Ok(nodal_forces)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<NodalStiffnessesBlock, ConstitutiveError> {
        let mut nodal_stiffnesses = NodalStiffnessesBlock::zero(nodal_coordinates.len());
        self.elements()
            .iter()
            .zip(self.connectivity().iter())
            .try_for_each(|(element, element_connectivity)| {
                element
                    .nodal_stiffnesses(
                        &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                    )?
                    .iter()
                    .zip(element_connectivity.iter())
                    .for_each(|(object, &node_a)| {
                        object.iter().zip(element_connectivity.iter()).for_each(
                            |(nodal_stiffness, &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                            },
                        )
                    });
                Ok::<(), ConstitutiveError>(())
            })?;
        Ok(nodal_stiffnesses)
    }
    fn root(
        &self,
        initial_coordinates: NodalCoordinatesBlock,
        root_finding: NewtonRaphson,
        equality_constraint: EqualityConstraint,
    ) -> Result<NodalCoordinatesBlock, OptimizeError> {
        root_finding.root(
            |nodal_coordinates: &NodalCoordinatesBlock| Ok(self.nodal_forces(nodal_coordinates)?),
            |nodal_coordinates: &NodalCoordinatesBlock| {
                Ok(self.nodal_stiffnesses(nodal_coordinates)?)
            },
            initial_coordinates,
            equality_constraint,
        )
    }
}

impl<C, F, const G: usize, const N: usize> HyperelasticFiniteElementBlock<C, F, G, N>
    for ElementBlock<F, N>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, N>,
    Self: ElasticFiniteElementBlock<C, F, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<Scalar, ConstitutiveError> {
        self.elements()
            .iter()
            .zip(self.connectivity().iter())
            .map(|(element, element_connectivity)| {
                element.helmholtz_free_energy(
                    &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                )
            })
            .sum()
    }
    fn minimize(
        &self,
        initial_coordinates: NodalCoordinatesBlock,
        optimization: NewtonRaphson,
        equality_constraint: EqualityConstraint,
    ) -> Result<NodalCoordinatesBlock, OptimizeError> {
        //
        //
        let banded = band(self.connectivity(), &equality_constraint, initial_coordinates.len());
        //
        //
        optimization.minimize(
            |nodal_coordinates: &NodalCoordinatesBlock| {
                Ok(self.helmholtz_free_energy(nodal_coordinates)?)
            },
            |nodal_coordinates: &NodalCoordinatesBlock| Ok(self.nodal_forces(nodal_coordinates)?),
            |nodal_coordinates: &NodalCoordinatesBlock| {
                Ok(self.nodal_stiffnesses(nodal_coordinates)?)
            },
            initial_coordinates,
            equality_constraint,
            Some(banded),
        )
    }
}

impl<C, F, const G: usize, const N: usize> ViscoelasticFiniteElementBlock<C, F, G, N>
    for ElementBlock<F, N>
where
    C: Viscoelastic,
    F: ViscoelasticFiniteElement<C, G, N>,
    Self: FiniteElementBlockMethods<C, F, G, N>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        nodal_velocities: &NodalVelocitiesBlock,
    ) -> Result<NodalForcesBlock, ConstitutiveError> {
        let mut nodal_forces = NodalForcesBlock::zero(nodal_coordinates.len());
        self.elements()
            .iter()
            .zip(self.connectivity().iter())
            .try_for_each(|(element, element_connectivity)| {
                element
                    .nodal_forces(
                        &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                        &self.nodal_velocities_element(element_connectivity, nodal_velocities),
                    )?
                    .iter()
                    .zip(element_connectivity.iter())
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), ConstitutiveError>(())
            })?;
        Ok(nodal_forces)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        nodal_velocities: &NodalVelocitiesBlock,
    ) -> Result<NodalStiffnessesBlock, ConstitutiveError> {
        let mut nodal_stiffnesses = NodalStiffnessesBlock::zero(nodal_coordinates.len());
        self.elements()
            .iter()
            .zip(self.connectivity().iter())
            .try_for_each(|(element, element_connectivity)| {
                element
                    .nodal_stiffnesses(
                        &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                        &self.nodal_velocities_element(element_connectivity, nodal_velocities),
                    )?
                    .iter()
                    .zip(element_connectivity.iter())
                    .for_each(|(object, &node_a)| {
                        object.iter().zip(element_connectivity.iter()).for_each(
                            |(nodal_stiffness, &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                            },
                        )
                    });
                Ok::<(), ConstitutiveError>(())
            })?;
        Ok(nodal_stiffnesses)
    }
    fn nodal_velocities_element(
        &self,
        element_connectivity: &[usize; N],
        nodal_velocities: &NodalVelocitiesBlock,
    ) -> NodalVelocities<N> {
        element_connectivity
            .iter()
            .map(|node| nodal_velocities[*node].clone())
            .collect()
    }
}

impl<C, F, const G: usize, const N: usize> ElasticHyperviscousFiniteElementBlock<C, F, G, N>
    for ElementBlock<F, N>
where
    C: ElasticHyperviscous,
    F: ElasticHyperviscousFiniteElement<C, G, N>,
    Self: ViscoelasticFiniteElementBlock<C, F, G, N>,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        nodal_velocities: &NodalVelocitiesBlock,
    ) -> Result<Scalar, ConstitutiveError> {
        self.elements()
            .iter()
            .zip(self.connectivity().iter())
            .map(|(element, element_connectivity)| {
                element.viscous_dissipation(
                    &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                    &self.nodal_velocities_element(element_connectivity, nodal_velocities),
                )
            })
            .sum()
    }
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        nodal_velocities: &NodalVelocitiesBlock,
    ) -> Result<Scalar, ConstitutiveError> {
        self.elements()
            .iter()
            .zip(self.connectivity().iter())
            .map(|(element, element_connectivity)| {
                element.dissipation_potential(
                    &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                    &self.nodal_velocities_element(element_connectivity, nodal_velocities),
                )
            })
            .sum()
    }
}

impl<C, F, const G: usize, const N: usize> HyperviscoelasticFiniteElementBlock<C, F, G, N>
    for ElementBlock<F, N>
where
    C: Hyperviscoelastic,
    F: HyperviscoelasticFiniteElement<C, G, N>,
    Self: ElasticHyperviscousFiniteElementBlock<C, F, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<Scalar, ConstitutiveError> {
        self.elements()
            .iter()
            .zip(self.connectivity().iter())
            .map(|(element, element_connectivity)| {
                element.helmholtz_free_energy(
                    &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                )
            })
            .sum()
    }
}

fn band<const N: usize>(connectivity: &Connectivity<N>, equality_constraint: &EqualityConstraint, number_of_nodes: usize) -> Banded {
    match equality_constraint {
        EqualityConstraint::Linear(matrix, _) => {
            let neighbors: Vec<Vec<usize>> = invert(connectivity, number_of_nodes)
                .iter()
                .map(|elements| {
                    let mut bar: Vec<usize> = elements
                        .iter()
                        .flat_map(|&element| connectivity[element])
                        .collect();
                    bar.sort();
                    bar.dedup();
                    bar
                })
                .collect();
            let structure: Vec<Vec<bool>> = neighbors
                .iter()
                .map(|nodes| (0..number_of_nodes).map(|b| nodes.contains(&b)).collect())
                .collect();
            let foo: Vec<Vec<bool>> = structure.iter().flat_map(|row|
                repeat_n(row.iter().flat_map(|entry|
                    repeat_n(*entry, 3)
                ).collect(), 3)
            ).collect();
            let num_coords = 3 * number_of_nodes;
            assert_eq!(matrix.width(), num_coords);
            let num_dof = matrix.len() + matrix.width();
            let mut bar = vec![vec![false; num_dof]; num_dof];
            foo.iter().zip(bar.iter_mut()).for_each(|(foo_i, bar_i)|
                foo_i.iter().zip(bar_i.iter_mut()).for_each(|(foo_ij, bar_ij)|
                    *bar_ij = *foo_ij
                )
            );
            let mut index = num_coords;
            matrix.iter().for_each(|matrix_i|
                matrix_i.iter().enumerate().for_each(|(j, matrix_ij)|
                    if matrix_ij != &0.0 {
                        bar[index][j] = true;
                        bar[j][index] = true;
                        index += 1;
                    }
                )
            );
            Banded::from(bar)
        }
        _ => unimplemented!(),
    }
}

fn invert<const N: usize>(connectivity: &Connectivity<N>, number_of_nodes: usize) -> Vec<Vec<usize>> {
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

// fn permute_new<const N: usize>(connectivity: &Connectivity<N>, matrix: &Matrix, number_of_nodes: usize) -> Vec<usize> {
//     let mut inverse_connectivity = vec![vec![]; number_of_nodes];
//     connectivity
//         .iter()
//         .enumerate()
//         .for_each(|(element, nodes)| {
//             nodes
//                 .iter()
//                 .for_each(|&node| inverse_connectivity[node].push(element))
//         });
//     let neighbors: Vec<Vec<usize>> = inverse_connectivity
//         .iter()
//         .map(|elements| {
//             let mut bar: Vec<usize> = elements
//                 .iter()
//                 .flat_map(|&element| connectivity[element])
//                 .collect();
//             bar.sort();
//             bar.dedup();
//             bar
//         })
//         .collect();
//     let structure: Vec<Vec<bool>> = neighbors
//         .iter()
//         .map(|nodes| (0..number_of_nodes).map(|b| nodes.contains(&b)).collect())
//         .collect();
//     let foo: Vec<Vec<bool>> = structure.iter().flat_map(|row|
//         repeat_n(row.iter().flat_map(|entry|
//             repeat_n(*entry, 3)
//         ).collect(), 3)
//     ).collect();
//     let num_coords = 3 * number_of_nodes;
//     assert_eq!(matrix.len(), num_coords);
//     let num_dof = matrix.len() + matrix.width();
//     let mut bar = vec![vec![false; num_dof]; num_dof];
//     foo.iter().zip(bar.iter_mut()).for_each(|(foo_i, bar_i)|
//         foo_i.iter().zip(bar_i.iter_mut()).for_each(|(foo_ij, bar_ij)|
//             *bar_ij = *foo_ij
//         )
//     );
//     let mut index = num_coords;
//     matrix.iter().for_each(|matrix_i|
//         matrix_i.iter().enumerate().for_each(|(j, matrix_ij)|
//             if matrix_ij != &0.0 {
//                 bar[index][j] = true;
//                 bar[j][index] = true;
//                 index += 1;
//             }
//         )
//     );
//     // Step 1: Convert binary matrix to adjacency list
//     let mut adj_list = vec![Vec::new(); num_dof];
//     for i in 0..num_dof {
//         for j in 0..num_dof {
//             if structure[i][j] && i != j {
//                 adj_list[i].push(j);
//             }
//         }
//     }
//     // Step 2: Find the starting vertex (vertex with the smallest degree)
//     let start_vertex = (0..num_dof)
//         .min_by_key(|&i| adj_list[i].len())
//         .expect("Matrix must have at least one vertex");
//     // Step 3: Perform Breadth-First Search (BFS)
//     let mut visited = vec![false; num_dof]; // Track visited vertices
//     let mut order = Vec::new(); // Final reordered indices
//     let mut queue = VecDeque::new();
//     queue.push_back(start_vertex);
//     visited[start_vertex] = true;
//     while let Some(vertex) = queue.pop_front() {
//         order.push(vertex);

//         // Get neighbors of the current vertex, sorted by degree (ascending)
//         let mut neighbors: Vec<usize> = adj_list[vertex]
//             .iter()
//             .filter(|&&neighbor| !visited[neighbor])
//             .copied()
//             .collect();
//         neighbors.sort_by_key(|&neighbor| adj_list[neighbor].len());

//         // Add unvisited neighbors to the queue
//         for neighbor in neighbors {
//             visited[neighbor] = true;
//             queue.push_back(neighbor);
//         }
//     }
//     order.reverse();
//     order
// }

// fn permute<const N: usize>(connectivity: &Connectivity<N>, number_of_nodes: usize) -> Vec<usize> {
//     let mut inverse_connectivity = vec![vec![]; number_of_nodes];
//     connectivity
//         .iter()
//         .enumerate()
//         .for_each(|(element, nodes)| {
//             nodes
//                 .iter()
//                 .for_each(|&node| inverse_connectivity[node].push(element))
//         });
//     let neighbors: Vec<Vec<usize>> = inverse_connectivity
//         .iter()
//         .map(|elements| {
//             let mut bar: Vec<usize> = elements
//                 .iter()
//                 .flat_map(|&element| connectivity[element])
//                 .collect();
//             bar.sort();
//             bar.dedup();
//             bar
//         })
//         .collect();
//     let structure: Vec<Vec<bool>> = neighbors
//         .iter()
//         .map(|nodes| (0..number_of_nodes).map(|b| nodes.contains(&b)).collect())
//         .collect();
//     assert!(structure.iter().all(|row| row.len() == number_of_nodes));
//     // Step 1: Convert binary matrix to adjacency list
//     let mut adj_list = vec![Vec::new(); number_of_nodes];
//     for i in 0..number_of_nodes {
//         for j in 0..number_of_nodes {
//             if structure[i][j] && i != j {
//                 adj_list[i].push(j);
//             }
//         }
//     }
//     // Step 2: Find the starting vertex (vertex with the smallest degree)
//     let start_vertex = (0..number_of_nodes)
//         .min_by_key(|&i| adj_list[i].len())
//         .expect("Matrix must have at least one vertex");
//     // Step 3: Perform Breadth-First Search (BFS)
//     let mut visited = vec![false; number_of_nodes]; // Track visited vertices
//     let mut order = Vec::new(); // Final reordered indices
//     let mut queue = VecDeque::new();
//     queue.push_back(start_vertex);
//     visited[start_vertex] = true;
//     while let Some(vertex) = queue.pop_front() {
//         order.push(vertex);

//         // Get neighbors of the current vertex, sorted by degree (ascending)
//         let mut neighbors: Vec<usize> = adj_list[vertex]
//             .iter()
//             .filter(|&&neighbor| !visited[neighbor])
//             .copied()
//             .collect();
//         neighbors.sort_by_key(|&neighbor| adj_list[neighbor].len());

//         // Add unvisited neighbors to the queue
//         for neighbor in neighbors {
//             visited[neighbor] = true;
//             queue.push_back(neighbor);
//         }
//     }
//     order.reverse();
//     order
// }
