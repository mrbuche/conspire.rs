#[cfg(test)]
mod test;

pub mod element;

use self::element::{
    ElasticFiniteElement, ElasticHyperviscousFiniteElement, FiniteElement, FiniteElementMethods,
    HyperelasticFiniteElement, HyperviscoelasticFiniteElement, SurfaceFiniteElement,
    ViscoelasticFiniteElement,
};
use super::*;
use crate::math::optimize::{Dirichlet, FirstOrder, GradientDescent, OptimizeError};
use std::array::from_fn;

pub struct ElementBlock<F, const N: usize> {
    connectivity: Connectivity<N>,
    elements: Vec<F>,
}

pub trait FiniteElementBlockMethods<'a, C, F, const G: usize, const N: usize, Y>
where
    C: Constitutive<Y>,
    F: FiniteElementMethods<'a, C, G, N>,
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

pub trait FiniteElementBlock<'a, C, F, const G: usize, const N: usize, Y>
where
    C: Constitutive<Y>,
    F: FiniteElement<'a, C, G, N>,
{
    fn new(
        constitutive_model_parameters: Y,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: ReferenceNodalCoordinatesBlock,
    ) -> Self;
}

pub trait SurfaceFiniteElementBlock<'a, C, F, const G: usize, const N: usize, const P: usize, Y>
where
    C: Constitutive<Y>,
    F: SurfaceFiniteElement<'a, C, G, N, P>,
{
    fn new(
        constitutive_model_parameters: Y,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: ReferenceNodalCoordinatesBlock,
        thickness: Scalar,
    ) -> Self;
}

impl<'a, C, F, const G: usize, const N: usize, Y> FiniteElementBlockMethods<'a, C, F, G, N, Y>
    for ElementBlock<F, N>
where
    C: Constitutive<Y>,
    F: FiniteElementMethods<'a, C, G, N>,
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

impl<'a, C, F, const G: usize, const N: usize, Y> FiniteElementBlock<'a, C, F, G, N, Y>
    for ElementBlock<F, N>
where
    C: Constitutive<Y>,
    F: FiniteElement<'a, C, G, N>,
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

impl<'a, C, F, const G: usize, const N: usize, const P: usize, Y>
    SurfaceFiniteElementBlock<'a, C, F, G, N, P, Y> for ElementBlock<F, N>
where
    C: Constitutive<Y>,
    F: SurfaceFiniteElement<'a, C, G, N, P>,
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

pub trait ElasticFiniteElementBlock<'a, C, F, const G: usize, const N: usize, Y>
where
    C: Elastic<Y>,
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

pub trait HyperelasticFiniteElementBlock<'a, C, F, const G: usize, const N: usize, Y>
where
    C: Hyperelastic<Y>,
    F: HyperelasticFiniteElement<'a, C, G, N>,
    Self: ElasticFiniteElementBlock<'a, C, F, G, N, Y>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<Scalar, ConstitutiveError>;
}

pub trait ViscoelasticFiniteElementBlock<'a, C, F, const G: usize, const N: usize>
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

pub trait ElasticHyperviscousFiniteElementBlock<'a, C, F, const G: usize, const N: usize, Y>
where
    C: ElasticHyperviscous<Y>,
    F: ElasticHyperviscousFiniteElement<'a, C, G, N>,
    Self: ViscoelasticFiniteElementBlock<'a, C, F, G, N, Y>,
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

pub trait HyperviscoelasticFiniteElementBlock<'a, C, F, const G: usize, const N: usize, Y>
where
    C: Hyperviscoelastic<Y>,
    F: HyperviscoelasticFiniteElement<'a, C, G, N>,
    Self: ElasticHyperviscousFiniteElementBlock<'a, C, F, G, N, Y>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<Scalar, ConstitutiveError>;
}

impl<'a, C, F, const G: usize, const N: usize, Y> ElasticFiniteElementBlock<'a, C, F, G, N, Y>
    for ElementBlock<F, N>
where
    C: Elastic<Y>,
    F: ElasticFiniteElement<'a, C, G, N>,
    Self: FiniteElementBlockMethods<'a, C, F, G, N, Y>,
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

impl<'a, C, F, const G: usize, const N: usize, Y> HyperelasticFiniteElementBlock<'a, C, F, G, N, Y>
    for ElementBlock<F, N>
where
    C: Hyperelastic<Y>,
    F: HyperelasticFiniteElement<'a, C, G, N>,
    Self: ElasticFiniteElementBlock<'a, C, F, G, N, Y>,
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

impl<'a, C, F, const G: usize, const N: usize, Y> ViscoelasticFiniteElementBlock<'a, C, F, G, N, Y>
    for ElementBlock<F, N>
where
    C: Viscoelastic<Y>,
    F: ViscoelasticFiniteElement<'a, C, G, N>,
    Self: FiniteElementBlockMethods<'a, C, F, G, N, Y>,
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

impl<'a, C, F, const G: usize, const N: usize, Y> ElasticHyperviscousFiniteElementBlock<'a, C, F, G, N, Y>
    for ElementBlock<F, N>
where
    C: ElasticHyperviscous<Y>,
    F: ElasticHyperviscousFiniteElement<'a, C, G, N>,
    Self: ViscoelasticFiniteElementBlock<'a, C, F, G, N, Y>,
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

impl<'a, C, F, const G: usize, const N: usize, Y> HyperviscoelasticFiniteElementBlock<'a, C, F, G, N, Y>
    for ElementBlock<F, N>
where
    C: Hyperviscoelastic<Y>,
    F: HyperviscoelasticFiniteElement<'a, C, G, N>,
    Self: ElasticHyperviscousFiniteElementBlock<'a, C, F, G, N, Y>,
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
