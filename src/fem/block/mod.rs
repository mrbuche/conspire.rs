#[cfg(test)]
mod test;

pub mod element;

use self::element::{
    ElasticFiniteElement, ElasticHyperviscousFiniteElement, FiniteElement, FiniteElementMethods,
    HyperelasticFiniteElement, HyperviscoelasticFiniteElement, SurfaceFiniteElement,
    ViscoelasticFiniteElement,
};
use super::*;
use crate::{
    math::{
        Banded,
        integrate::{Explicit, IntegrationError},
        optimize::{
            EqualityConstraint, FirstOrderRootFinding, OptimizeError, SecondOrderOptimization,
            ZerothOrderRootFinding,
        },
    },
    mechanics::Times,
};
use std::{array::from_fn, iter::repeat_n};

pub struct ElementBlock<F, const N: usize> {
    connectivity: Connectivity<N>,
    coordinates: ReferenceNodalCoordinatesBlock,
    elements: Vec<F>,
}

pub trait FiniteElementBlockMethods<C, F, const G: usize, const N: usize>
where
    F: FiniteElementMethods<C, G, N>,
{
    fn connectivity(&self) -> &Connectivity<N>;
    fn coordinates(&self) -> &ReferenceNodalCoordinatesBlock;
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Vec<DeformationGradientList<G>>;
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
    fn reset(&mut self);
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
    fn coordinates(&self) -> &ReferenceNodalCoordinatesBlock {
        &self.coordinates
    }
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Vec<DeformationGradientList<G>> {
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
        coordinates: ReferenceNodalCoordinatesBlock,
    ) -> Self {
        let elements = connectivity
            .iter()
            .map(|element_connectivity| {
                <F>::new(
                    constitutive_model_parameters,
                    element_connectivity
                        .iter()
                        .map(|&node| coordinates[node].clone())
                        .collect(),
                )
            })
            .collect();
        Self {
            connectivity,
            coordinates,
            elements,
        }
    }
    fn reset(&mut self) {
        self.elements.iter_mut().for_each(|element| element.reset())
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
        coordinates: ReferenceNodalCoordinatesBlock,
        thickness: Scalar,
    ) -> Self {
        let elements = connectivity
            .iter()
            .map(|element_connectivity| {
                <F>::new(
                    constitutive_model_parameters,
                    element_connectivity
                        .iter()
                        .map(|node| coordinates[*node].clone())
                        .collect(),
                    &thickness,
                )
            })
            .collect();
        Self {
            connectivity,
            coordinates,
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
}

pub trait ZerothOrderRoot<C, F, const G: usize, const N: usize>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, N>,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl ZerothOrderRootFinding<NodalCoordinatesBlock>,
    ) -> Result<NodalCoordinatesBlock, OptimizeError>;
}

pub trait FirstOrderRoot<C, F, const G: usize, const N: usize>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, N>,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderRootFinding<
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
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
}

pub trait SecondOrderMinimize<C, F, const G: usize, const N: usize>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, N>,
    Self: ElasticFiniteElementBlock<C, F, G, N>,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
    ) -> Result<NodalCoordinatesBlock, OptimizeError>;
}

pub trait ViscoelasticFiniteElementBlock<C, F, const G: usize, const N: usize>
where
    C: Viscoelastic,
    F: ViscoelasticFiniteElement<C, G, N>,
{
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        nodal_velocities: &NodalVelocitiesBlock,
    ) -> Vec<DeformationGradientRateList<G>>;
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
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        integrator: impl Explicit<NodalVelocitiesBlock, NodalVelocitiesHistory>,
        time: &[Scalar],
        solver: impl FirstOrderRootFinding<
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
    ) -> Result<(Times, NodalCoordinatesHistory, NodalVelocitiesHistory), IntegrationError>;
    #[doc(hidden)]
    fn root_inner(
        &self,
        equality_constraint: EqualityConstraint,
        nodal_coordinates: &NodalCoordinatesBlock,
        solver: &impl FirstOrderRootFinding<
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
        initial_guess: &NodalVelocitiesBlock,
    ) -> Result<NodalVelocitiesBlock, OptimizeError>;
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
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        integrator: impl Explicit<NodalVelocitiesBlock, NodalVelocitiesHistory>,
        time: &[Scalar],
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
    ) -> Result<(Times, NodalCoordinatesHistory, NodalVelocitiesHistory), IntegrationError>;
    #[doc(hidden)]
    fn minimize_inner(
        &self,
        equality_constraint: EqualityConstraint,
        nodal_coordinates: &NodalCoordinatesBlock,
        solver: &impl SecondOrderOptimization<
            Scalar,
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
        initial_guess: &NodalVelocitiesBlock,
    ) -> Result<NodalVelocitiesBlock, OptimizeError>;
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
}

// impl<C, F, const G: usize, const N: usize> ZerothOrderRoot<C, F, G, N> for ElementBlock<F, N>
// where
//     C: Elastic,
//     F: ElasticFiniteElement<C, G, N>,
//     Self: FiniteElementBlockMethods<C, F, G, N>,
// {
//     fn root(
//         &self,
//         equality_constraint: EqualityConstraint,
//         solver: impl ZerothOrderRootFinding<NodalCoordinatesBlock>,
//     ) -> Result<NodalCoordinatesBlock, OptimizeError> {
//         solver.root(
//             |nodal_coordinates: &NodalCoordinatesBlock| Ok(self.nodal_forces(nodal_coordinates)?),
//             self.coordinates().clone().into(),
//             equality_constraint,
//         )
//     }
// }

impl<C, F, const G: usize, const N: usize> FirstOrderRoot<C, F, G, N> for ElementBlock<F, N>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, N>,
    Self: FiniteElementBlockMethods<C, F, G, N>,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderRootFinding<
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
    ) -> Result<NodalCoordinatesBlock, OptimizeError> {
        solver.root(
            |nodal_coordinates: &NodalCoordinatesBlock| Ok(self.nodal_forces(nodal_coordinates)?),
            |nodal_coordinates: &NodalCoordinatesBlock| {
                Ok(self.nodal_stiffnesses(nodal_coordinates)?)
            },
            self.coordinates().clone().into(),
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
}

impl<C, F, const G: usize, const N: usize> SecondOrderMinimize<C, F, G, N> for ElementBlock<F, N>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, N>,
    Self: ElasticFiniteElementBlock<C, F, G, N>,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
    ) -> Result<NodalCoordinatesBlock, OptimizeError> {
        let banded = band(
            self.connectivity(),
            &equality_constraint,
            self.coordinates().len(),
        );
        solver.minimize(
            |nodal_coordinates: &NodalCoordinatesBlock| {
                Ok(self.helmholtz_free_energy(nodal_coordinates)?)
            },
            |nodal_coordinates: &NodalCoordinatesBlock| Ok(self.nodal_forces(nodal_coordinates)?),
            |nodal_coordinates: &NodalCoordinatesBlock| {
                Ok(self.nodal_stiffnesses(nodal_coordinates)?)
            },
            self.coordinates().clone().into(),
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
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        nodal_velocities: &NodalVelocitiesBlock,
    ) -> Vec<DeformationGradientRateList<G>> {
        self.elements()
            .iter()
            .zip(self.connectivity().iter())
            .map(|(element, element_connectivity)| {
                element.deformation_gradient_rates(
                    &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                    &self.nodal_velocities_element(element_connectivity, nodal_velocities),
                )
            })
            .collect()
    }
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
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        integrator: impl Explicit<NodalVelocitiesBlock, NodalVelocitiesHistory>,
        time: &[Scalar],
        solver: impl FirstOrderRootFinding<
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
    ) -> Result<(Times, NodalCoordinatesHistory, NodalVelocitiesHistory), IntegrationError> {
        let mut solution = NodalVelocitiesBlock::zero(self.coordinates().len());
        integrator.integrate(
            |_: Scalar, nodal_coordinates: &NodalCoordinatesBlock| {
                solution = self.root_inner(
                    equality_constraint.clone(),
                    nodal_coordinates,
                    &solver,
                    &solution,
                )?;
                Ok(solution.clone())
            },
            time,
            self.coordinates().clone().into(),
        )
    }
    #[doc(hidden)]
    fn root_inner(
        &self,
        equality_constraint: EqualityConstraint,
        nodal_coordinates: &NodalCoordinatesBlock,
        solver: &impl FirstOrderRootFinding<
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
        initial_guess: &NodalVelocitiesBlock,
    ) -> Result<NodalVelocitiesBlock, OptimizeError> {
        solver.root(
            |nodal_velocities: &NodalVelocitiesBlock| {
                Ok(self.nodal_forces(nodal_coordinates, nodal_velocities)?)
            },
            |nodal_velocities: &NodalVelocitiesBlock| {
                Ok(self.nodal_stiffnesses(nodal_coordinates, nodal_velocities)?)
            },
            initial_guess.clone(),
            equality_constraint,
        )
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
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        integrator: impl Explicit<NodalVelocitiesBlock, NodalVelocitiesHistory>,
        time: &[Scalar],
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
    ) -> Result<(Times, NodalCoordinatesHistory, NodalVelocitiesHistory), IntegrationError> {
        let mut solution = NodalVelocitiesBlock::zero(self.coordinates().len());
        integrator.integrate(
            |_: Scalar, nodal_coordinates: &NodalCoordinatesBlock| {
                solution = self.minimize_inner(
                    equality_constraint.clone(),
                    nodal_coordinates,
                    &solver,
                    &solution,
                )?;
                Ok(solution.clone())
            },
            time,
            self.coordinates().clone().into(),
        )
    }
    #[doc(hidden)]
    fn minimize_inner(
        &self,
        equality_constraint: EqualityConstraint,
        nodal_coordinates: &NodalCoordinatesBlock,
        solver: &impl SecondOrderOptimization<
            Scalar,
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
        initial_guess: &NodalVelocitiesBlock,
    ) -> Result<NodalVelocitiesBlock, OptimizeError> {
        let num_coords = nodal_coordinates.len();
        let banded = band(self.connectivity(), &equality_constraint, num_coords);
        solver.minimize(
            |nodal_velocities: &NodalVelocitiesBlock| {
                Ok(self.dissipation_potential(nodal_coordinates, nodal_velocities)?)
            },
            |nodal_velocities: &NodalVelocitiesBlock| {
                Ok(self.nodal_forces(nodal_coordinates, nodal_velocities)?)
            },
            |nodal_velocities: &NodalVelocitiesBlock| {
                Ok(self.nodal_stiffnesses(nodal_coordinates, nodal_velocities)?)
            },
            initial_guess.clone(),
            equality_constraint,
            Some(banded),
        )
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

fn band<const N: usize>(
    connectivity: &Connectivity<N>,
    equality_constraint: &EqualityConstraint,
    number_of_nodes: usize,
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
            let structure_3d: Vec<Vec<bool>> = structure
                .iter()
                .flat_map(|row| {
                    repeat_n(
                        row.iter().flat_map(|entry| repeat_n(*entry, 3)).collect(),
                        3,
                    )
                })
                .collect();
            let mut keep = vec![true; structure_3d.len()];
            indices.iter().for_each(|&index| keep[index] = false);
            let banded = structure_3d
                .into_iter()
                .zip(keep.iter())
                .filter(|(_, keep)| **keep)
                .map(|(structure_3d_a, _)| {
                    structure_3d_a
                        .into_iter()
                        .zip(keep.iter())
                        .filter(|(_, keep)| **keep)
                        .map(|(structure_3d_ab, _)| structure_3d_ab)
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
            let structure_3d: Vec<Vec<bool>> = structure
                .iter()
                .flat_map(|row| {
                    repeat_n(
                        row.iter().flat_map(|entry| repeat_n(*entry, 3)).collect(),
                        3,
                    )
                })
                .collect();
            let num_coords = 3 * number_of_nodes;
            assert_eq!(matrix.width(), num_coords);
            let num_dof = matrix.len() + matrix.width();
            let mut banded = vec![vec![false; num_dof]; num_dof];
            structure_3d
                .iter()
                .zip(banded.iter_mut())
                .for_each(|(structure_3d_i, banded_i)| {
                    structure_3d_i
                        .iter()
                        .zip(banded_i.iter_mut())
                        .for_each(|(structure_3d_ij, banded_ij)| *banded_ij = *structure_3d_ij)
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
            let structure_3d: Vec<Vec<bool>> = structure
                .iter()
                .flat_map(|row| {
                    repeat_n(
                        row.iter().flat_map(|entry| repeat_n(*entry, 3)).collect(),
                        3,
                    )
                })
                .collect();
            Banded::from(structure_3d)
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
