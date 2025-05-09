#[cfg(test)]
mod test;

pub mod element;

use self::element::{
    ElasticFiniteElement, ElasticHyperviscousFiniteElement, FiniteElement, FiniteElementMethods,
    HyperelasticFiniteElement, HyperviscoelasticFiniteElement, SurfaceFiniteElement,
    ViscoelasticFiniteElement,
};
use super::*;
use crate::math::optimize::{
    EqualityConstraint, FirstOrderRootFinding, NewtonRaphson, OptimizeError,
    SecondOrderOptimization,
};
use std::array::from_fn;

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
    // fn structure(&self, number_of_nodes: usize) -> Vec<Vec<bool>>;
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
    // fn structure(&self, number_of_nodes: usize) -> Vec<Vec<bool>> {
    //     let connectivity = self.connectivity();
    //     let mut inverse_connectivity = vec![vec![]; number_of_nodes];
    //     connectivity
    //         .iter()
    //         .enumerate()
    //         .for_each(|(element, nodes)| {
    //             nodes
    //                 .iter()
    //                 .for_each(|&node| inverse_connectivity[node].push(element))
    //         });
    //     let foo: Vec<Vec<usize>> = inverse_connectivity
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
    //     // now populate SquatrMatrix of bools, if that is the best structure to rearrange for something
    //     //
    //     // Cuthill–McKee algorithm (and reverse) permutes SYMMETRIC sparse matrices into a narrow band matrix.
    //     // Supposedly helps both aspects of performance, and also numerical stability, when solving equations.
    //     //
    //     // Seems there are Cholesky algorithms for banded Hermitian positive-definite matrices,
    //     // so this re-structuring might be a good idea after all.
    //     //
    //     todo!()
    // }
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
                        .map(|node| reference_nodal_coordinates[*node].clone())
                        .collect(),
                )
            })
            .collect();
        Self {
            connectivity,
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
