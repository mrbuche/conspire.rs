#[cfg(test)]
mod test;

pub mod element;

use self::element::{
    ElasticFiniteElement, ElasticHyperviscousFiniteElement, FiniteElement,
    HyperelasticFiniteElement, HyperviscoelasticFiniteElement, ViscoelasticFiniteElement,
};
use super::*;
use crate::math::optimize::{Dirichlet, FirstOrder, GradientDescent, OptimizeError};
use std::array::from_fn;

pub struct ElasticBlock<const E: usize, F, const N: usize> {
    connectivity: Connectivity<E, N>,
    elements: [F; E],
}

pub struct ViscoelasticBlock<const E: usize, F, const N: usize> {
    connectivity: Connectivity<E, N>,
    elements: [F; E],
}

pub trait BasicFiniteElementBlock<'a, C, const E: usize, F, const G: usize, const N: usize>
where
    C: Constitutive<'a>,
{
    fn connectivity(&self) -> &Connectivity<E, N>;
    fn elements(&self) -> &[F; E];
    fn nodal_coordinates_element(
        &self,
        element_connectivity: &[usize; N],
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> NodalCoordinates<N>;
}

pub trait FiniteElementBlock<'a, C, const E: usize, F, const G: usize, const N: usize>
where
    C: Constitutive<'a>,
    F: FiniteElement<'a, C, G, N>,
    Self: BasicFiniteElementBlock<'a, C, E, F, G, N>,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> DeformationGradientss<G, E> {
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
    fn new(
        constitutive_model_parameters: Parameters<'a>,
        connectivity: Connectivity<E, N>,
        reference_nodal_coordinates: ReferenceNodalCoordinatesBlock,
    ) -> Self;
}

pub trait ElasticFiniteElementBlock<'a, C, const E: usize, F, const G: usize, const N: usize>
where
    C: Elastic<'a>,
    F: ElasticFiniteElement<'a, C, G, N>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<NodalForcesBlock, ConstitutiveError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<NodalStiffnessesBlock, ConstitutiveError>;
    fn solve(
        &self,
        initial_coordinates: NodalCoordinatesBlock,
        places_d: Option<&[&[usize]]>,
        values_d: Option<&[Scalar]>,
        places_n: Option<&[&[usize]]>,
        values_n: Option<&[Scalar]>,
        optimization: GradientDescent,
    ) -> Result<NodalCoordinatesBlock, OptimizeError>;
}

pub trait HyperelasticFiniteElementBlock<'a, C, const E: usize, F, const G: usize, const N: usize>
where
    C: Hyperelastic<'a>,
    F: HyperelasticFiniteElement<'a, C, G, N>,
    Self: ElasticFiniteElementBlock<'a, C, E, F, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<Scalar, ConstitutiveError>;
}

pub trait ViscoelasticFiniteElementBlock<'a, C, const E: usize, F, const G: usize, const N: usize>
where
    C: Viscoelastic<'a>,
    F: ViscoelasticFiniteElement<'a, C, G, N>,
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

pub trait ElasticHyperviscousFiniteElementBlock<
    'a,
    C,
    const E: usize,
    F,
    const G: usize,
    const N: usize,
> where
    C: ElasticHyperviscous<'a>,
    F: ElasticHyperviscousFiniteElement<'a, C, G, N>,
    Self: ViscoelasticFiniteElementBlock<'a, C, E, F, G, N>,
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

pub trait HyperviscoelasticFiniteElementBlock<
    'a,
    C,
    const E: usize,
    F,
    const G: usize,
    const N: usize,
> where
    C: Hyperviscoelastic<'a>,
    F: HyperviscoelasticFiniteElement<'a, C, G, N>,
    Self: ElasticHyperviscousFiniteElementBlock<'a, C, E, F, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<Scalar, ConstitutiveError>;
}

impl<'a, C, const E: usize, F, const G: usize, const N: usize>
    BasicFiniteElementBlock<'a, C, E, F, G, N> for ElasticBlock<E, F, N>
where
    C: Elastic<'a>,
    F: ElasticFiniteElement<'a, C, G, N>,
{
    fn connectivity(&self) -> &Connectivity<E, N> {
        &self.connectivity
    }
    fn elements(&self) -> &[F; E] {
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

impl<'a, C, const E: usize, F, const G: usize, const N: usize> FiniteElementBlock<'a, C, E, F, G, N>
    for ElasticBlock<E, F, N>
where
    C: Elastic<'a>,
    F: FiniteElement<'a, C, G, N> + ElasticFiniteElement<'a, C, G, N>,
{
    fn new(
        constitutive_model_parameters: Parameters<'a>,
        connectivity: Connectivity<E, N>,
        reference_nodal_coordinates: ReferenceNodalCoordinatesBlock,
    ) -> Self {
        let elements = from_fn(|element| {
            <F>::new(
                constitutive_model_parameters,
                connectivity[element]
                    .iter()
                    .map(|node| reference_nodal_coordinates[*node].clone())
                    .collect(),
            )
        });
        Self {
            connectivity,
            elements,
        }
    }
}

impl<'a, C, const E: usize, F, const G: usize, const N: usize>
    ElasticFiniteElementBlock<'a, C, E, F, G, N> for ElasticBlock<E, F, N>
where
    C: Elastic<'a>,
    F: ElasticFiniteElement<'a, C, G, N>,
    Self: BasicFiniteElementBlock<'a, C, E, F, G, N>,
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
                    .for_each(|(nodal_force, node)| nodal_forces[*node] += nodal_force);
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
                    .for_each(|(object, node_a)| {
                        object.iter().zip(element_connectivity.iter()).for_each(
                            |(nodal_stiffness, node_b)| {
                                nodal_stiffnesses[*node_a][*node_b] += nodal_stiffness
                            },
                        )
                    });
                Ok::<(), ConstitutiveError>(())
            })?;
        Ok(nodal_stiffnesses)
    }
    fn solve(
        &self,
        initial_coordinates: NodalCoordinatesBlock,
        places_d: Option<&[&[usize]]>,
        values_d: Option<&[Scalar]>,
        _places_n: Option<&[&[usize]]>,
        _values_n: Option<&[Scalar]>,
        optimization: GradientDescent,
    ) -> Result<NodalCoordinatesBlock, OptimizeError> {
        optimization.minimize(
            |nodal_coordinates: &NodalCoordinatesBlock| Ok(self.nodal_forces(nodal_coordinates)?),
            initial_coordinates,
            Some(Dirichlet {
                places: places_d.unwrap(),
                values: values_d.unwrap(),
            }),
            None,
        )
    }
}

impl<'a, C, const E: usize, F, const G: usize, const N: usize>
    HyperelasticFiniteElementBlock<'a, C, E, F, G, N> for ElasticBlock<E, F, N>
where
    C: Hyperelastic<'a>,
    F: HyperelasticFiniteElement<'a, C, G, N>,
    Self: ElasticFiniteElementBlock<'a, C, E, F, G, N>,
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

impl<'a, C, const E: usize, F, const G: usize, const N: usize>
    BasicFiniteElementBlock<'a, C, E, F, G, N> for ViscoelasticBlock<E, F, N>
where
    C: Viscoelastic<'a>,
    F: ViscoelasticFiniteElement<'a, C, G, N>,
{
    fn connectivity(&self) -> &Connectivity<E, N> {
        &self.connectivity
    }
    fn elements(&self) -> &[F; E] {
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

impl<'a, C, const E: usize, F, const G: usize, const N: usize> FiniteElementBlock<'a, C, E, F, G, N>
    for ViscoelasticBlock<E, F, N>
where
    C: Viscoelastic<'a>,
    F: FiniteElement<'a, C, G, N> + ViscoelasticFiniteElement<'a, C, G, N>,
{
    fn new(
        constitutive_model_parameters: Parameters<'a>,
        connectivity: Connectivity<E, N>,
        reference_nodal_coordinates: ReferenceNodalCoordinatesBlock,
    ) -> Self {
        let elements = from_fn(|element| {
            <F>::new(
                constitutive_model_parameters,
                connectivity[element]
                    .iter()
                    .map(|node| reference_nodal_coordinates[*node].clone())
                    .collect(),
            )
        });
        Self {
            connectivity,
            elements,
        }
    }
}

impl<'a, C, const E: usize, F, const G: usize, const N: usize>
    ViscoelasticFiniteElementBlock<'a, C, E, F, G, N> for ViscoelasticBlock<E, F, N>
where
    C: Viscoelastic<'a>,
    F: ViscoelasticFiniteElement<'a, C, G, N>,
    Self: BasicFiniteElementBlock<'a, C, E, F, G, N>,
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
                    .for_each(|(nodal_force, node)| nodal_forces[*node] += nodal_force);
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
                    .for_each(|(object, node_a)| {
                        object.iter().zip(element_connectivity.iter()).for_each(
                            |(nodal_stiffness, node_b)| {
                                nodal_stiffnesses[*node_a][*node_b] += nodal_stiffness
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

impl<'a, C, const E: usize, F, const G: usize, const N: usize>
    ElasticHyperviscousFiniteElementBlock<'a, C, E, F, G, N> for ViscoelasticBlock<E, F, N>
where
    C: ElasticHyperviscous<'a>,
    F: ElasticHyperviscousFiniteElement<'a, C, G, N>,
    Self: ViscoelasticFiniteElementBlock<'a, C, E, F, G, N>,
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

impl<'a, C, const E: usize, F, const G: usize, const N: usize>
    HyperviscoelasticFiniteElementBlock<'a, C, E, F, G, N> for ViscoelasticBlock<E, F, N>
where
    C: Hyperviscoelastic<'a>,
    F: HyperviscoelasticFiniteElement<'a, C, G, N>,
    Self: ElasticHyperviscousFiniteElementBlock<'a, C, E, F, G, N>,
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
