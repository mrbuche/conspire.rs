#[cfg(test)]
mod test;

pub mod element;

use self::element::{
    ElasticFiniteElement, ElasticHyperviscousFiniteElement, ElasticViscoplasticFiniteElement,
    FiniteElement, FiniteElementError, FiniteElementMethods, HyperelasticFiniteElement,
    HyperviscoelasticFiniteElement, SurfaceFiniteElement, ViscoelasticFiniteElement,
};
use super::*;
use crate::{
    constitutive::solid::elastic_viscoplastic::ElasticViscoplastic,
    defeat_message,
    math::{
        Banded, TensorArray, TensorTupleListVec, TensorTupleListVec2D, TestError,
        integrate::{Explicit, ExplicitIV, IntegrationError},
        optimize::{
            EqualityConstraint, FirstOrderOptimization, FirstOrderRootFinding, OptimizationError,
            SecondOrderOptimization, ZerothOrderRootFinding,
        },
    },
    mechanics::{DeformationGradientPlastic, Times},
};
use std::{
    array::from_fn,
    fmt::{self, Debug, Display, Formatter},
    iter::repeat_n,
};

pub struct ElementBlock<F, const N: usize> {
    connectivity: Connectivity<N>,
    coordinates: ReferenceNodalCoordinatesBlock,
    elements: Vec<F>,
}

impl<F, const N: usize> Debug for ElementBlock<F, N> {
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
            "ElementBlock {{ elements: [{element}; {}] }}",
            self.connectivity.len()
        )
    }
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

pub trait FiniteElementBlock<'a, C, F, const G: usize, const N: usize>
where
    F: FiniteElement<'a, C, G, N>,
{
    fn new(
        constitutive_model: &'a C,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: ReferenceNodalCoordinatesBlock,
    ) -> Self;
    fn reset(&mut self);
}

pub trait SurfaceFiniteElementBlock<'a, C, F, const G: usize, const N: usize, const P: usize>
where
    F: SurfaceFiniteElement<'a, C, G, N, P>,
{
    fn new(
        constitutive_model: &'a C,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: ReferenceNodalCoordinatesBlock,
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

impl<'a, C, F, const G: usize, const N: usize> FiniteElementBlock<'a, C, F, G, N>
    for ElementBlock<F, N>
where
    F: FiniteElement<'a, C, G, N>,
{
    fn new(
        constitutive_model: &'a C,
        connectivity: Connectivity<N>,
        coordinates: ReferenceNodalCoordinatesBlock,
    ) -> Self {
        let elements = connectivity
            .iter()
            .map(|element_connectivity| {
                <F>::new(
                    constitutive_model,
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

impl<'a, C, F, const G: usize, const N: usize, const P: usize>
    SurfaceFiniteElementBlock<'a, C, F, G, N, P> for ElementBlock<F, N>
where
    F: SurfaceFiniteElement<'a, C, G, N, P>,
{
    fn new(
        constitutive_model: &'a C,
        connectivity: Connectivity<N>,
        coordinates: ReferenceNodalCoordinatesBlock,
        thickness: Scalar,
    ) -> Self {
        let elements = connectivity
            .iter()
            .map(|element_connectivity| {
                <F>::new(
                    constitutive_model,
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
    ) -> Result<NodalForcesBlock, FiniteElementBlockError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<NodalStiffnessesBlock, FiniteElementBlockError>;
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
    ) -> Result<NodalCoordinatesBlock, OptimizationError>;
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
    ) -> Result<NodalCoordinatesBlock, OptimizationError>;
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
    ) -> Result<Scalar, FiniteElementBlockError>;
}

pub trait FirstOrderMinimize<C, F, const G: usize, const N: usize>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, N>,
    Self: ElasticFiniteElementBlock<C, F, G, N>,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderOptimization<Scalar, NodalForcesBlock>,
    ) -> Result<NodalCoordinatesBlock, OptimizationError>;
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
    ) -> Result<NodalCoordinatesBlock, OptimizationError>;
}

pub type ViscoplasticStateVariables<const G: usize> =
    TensorTupleListVec<DeformationGradientPlastic, Scalar, G>;

pub type ViscoplasticStateVariablesHistory<const G: usize> =
    TensorTupleListVec2D<DeformationGradientPlastic, Scalar, G>;

pub type Foo = (crate::math::Matrix, fn(Scalar) -> crate::math::Vector);

pub trait ElasticViscoplasticFiniteElementBlock<C, F, const G: usize, const N: usize>
where
    C: ElasticViscoplastic,
    F: ElasticViscoplasticFiniteElement<C, G, N>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<NodalForcesBlock, FiniteElementBlockError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<NodalStiffnessesBlock, FiniteElementBlockError>;
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<ViscoplasticStateVariables<G>, FiniteElementBlockError>;
    fn root(
        &self,
        foo: Foo,
        integrator: impl ExplicitIV<
            ViscoplasticStateVariables<G>,
            NodalCoordinatesBlock,
            ViscoplasticStateVariablesHistory<G>,
            NodalCoordinatesHistory,
        >,
        time: &[Scalar],
        solver: impl FirstOrderRootFinding<
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
    ) -> Result<
        (
            Times,
            NodalCoordinatesHistory,
            ViscoplasticStateVariablesHistory<G>,
        ),
        IntegrationError,
    >;
    #[doc(hidden)]
    fn root_inner(
        &self,
        equality_constraint: EqualityConstraint,
        state_variables: &ViscoplasticStateVariables<G>,
        solver: &impl FirstOrderRootFinding<
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
        initial_guess: &NodalCoordinatesBlock,
    ) -> Result<NodalCoordinatesBlock, OptimizationError>;
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
    ) -> Result<NodalForcesBlock, FiniteElementBlockError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        nodal_velocities: &NodalVelocitiesBlock,
    ) -> Result<NodalStiffnessesBlock, FiniteElementBlockError>;
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
        equality_constraint: &EqualityConstraint,
        nodal_coordinates: &NodalCoordinatesBlock,
        solver: &impl FirstOrderRootFinding<
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
        initial_guess: &NodalVelocitiesBlock,
    ) -> Result<NodalVelocitiesBlock, OptimizationError>;
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
    ) -> Result<Scalar, FiniteElementBlockError>;
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        nodal_velocities: &NodalVelocitiesBlock,
    ) -> Result<Scalar, FiniteElementBlockError>;
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
        equality_constraint: &EqualityConstraint,
        nodal_coordinates: &NodalCoordinatesBlock,
        solver: &impl SecondOrderOptimization<
            Scalar,
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
        initial_guess: &NodalVelocitiesBlock,
    ) -> Result<NodalVelocitiesBlock, OptimizationError>;
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
    ) -> Result<Scalar, FiniteElementBlockError>;
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
    ) -> Result<NodalForcesBlock, FiniteElementBlockError> {
        let mut nodal_forces = NodalForcesBlock::zero(nodal_coordinates.len());
        match self
            .elements()
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
                Ok::<(), FiniteElementError>(())
            }) {
            Ok(()) => Ok(nodal_forces),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<NodalStiffnessesBlock, FiniteElementBlockError> {
        let mut nodal_stiffnesses = NodalStiffnessesBlock::zero(nodal_coordinates.len());
        match self
            .elements()
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
                Ok::<(), FiniteElementError>(())
            }) {
            Ok(()) => Ok(nodal_stiffnesses),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<C, F, const G: usize, const N: usize> ZerothOrderRoot<C, F, G, N> for ElementBlock<F, N>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, N>,
    Self: FiniteElementBlockMethods<C, F, G, N>,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl ZerothOrderRootFinding<NodalCoordinatesBlock>,
    ) -> Result<NodalCoordinatesBlock, OptimizationError> {
        solver.root(
            |nodal_coordinates: &NodalCoordinatesBlock| Ok(self.nodal_forces(nodal_coordinates)?),
            self.coordinates().clone().into(),
            equality_constraint,
        )
    }
}

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
    ) -> Result<NodalCoordinatesBlock, OptimizationError> {
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
    ) -> Result<Scalar, FiniteElementBlockError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity().iter())
            .map(|(element, element_connectivity)| {
                element.helmholtz_free_energy(
                    &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                )
            })
            .sum()
        {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<C, F, const G: usize, const N: usize> FirstOrderMinimize<C, F, G, N> for ElementBlock<F, N>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, N>,
    Self: ElasticFiniteElementBlock<C, F, G, N>,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderOptimization<Scalar, NodalForcesBlock>,
    ) -> Result<NodalCoordinatesBlock, OptimizationError> {
        solver.minimize(
            |nodal_coordinates: &NodalCoordinatesBlock| {
                Ok(self.helmholtz_free_energy(nodal_coordinates)?)
            },
            |nodal_coordinates: &NodalCoordinatesBlock| Ok(self.nodal_forces(nodal_coordinates)?),
            self.coordinates().clone().into(),
            equality_constraint,
        )
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
    ) -> Result<NodalCoordinatesBlock, OptimizationError> {
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

impl<C, F, const G: usize, const N: usize> ElasticViscoplasticFiniteElementBlock<C, F, G, N>
    for ElementBlock<F, N>
where
    C: ElasticViscoplastic,
    F: ElasticViscoplasticFiniteElement<C, G, N>,
    Self: FiniteElementBlockMethods<C, F, G, N>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<NodalForcesBlock, FiniteElementBlockError> {
        let mut nodal_forces = NodalForcesBlock::zero(nodal_coordinates.len());
        match self
            .elements()
            .iter()
            .zip(self.connectivity().iter())
            .zip(state_variables.iter())
            .try_for_each(
                |((element, element_connectivity), state_variables_element)| {
                    element
                        .nodal_forces(
                            &self
                                .nodal_coordinates_element(element_connectivity, nodal_coordinates),
                            state_variables_element,
                        )?
                        .iter()
                        .zip(element_connectivity.iter())
                        .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                    Ok::<(), FiniteElementError>(())
                },
            ) {
            Ok(()) => Ok(nodal_forces),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<NodalStiffnessesBlock, FiniteElementBlockError> {
        let mut nodal_stiffnesses = NodalStiffnessesBlock::zero(nodal_coordinates.len());
        match self
            .elements()
            .iter()
            .zip(self.connectivity().iter())
            .zip(state_variables.iter())
            .try_for_each(
                |((element, element_connectivity), state_variables_element)| {
                    element
                        .nodal_stiffnesses(
                            &self
                                .nodal_coordinates_element(element_connectivity, nodal_coordinates),
                            state_variables_element,
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
                    Ok::<(), FiniteElementError>(())
                },
            ) {
            Ok(()) => Ok(nodal_stiffnesses),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<ViscoplasticStateVariables<G>, FiniteElementBlockError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity().iter())
            .zip(state_variables.iter())
            .map(
                |((element, element_connectivity), element_state_variables)| {
                    element.state_variables_evolution(
                        &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                        element_state_variables,
                    )
                },
            )
            .collect()
        {
            Ok(state_variables_evolution) => Ok(state_variables_evolution),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn root(
        &self,
        foo: Foo,
        integrator: impl ExplicitIV<
            ViscoplasticStateVariables<G>,
            NodalCoordinatesBlock,
            ViscoplasticStateVariablesHistory<G>,
            NodalCoordinatesHistory,
        >,
        time: &[Scalar],
        solver: impl FirstOrderRootFinding<
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
    ) -> Result<
        (
            Times,
            NodalCoordinatesHistory,
            ViscoplasticStateVariablesHistory<G>,
        ),
        IntegrationError,
    > {
        let (time_history, state_variables_history, _, nodal_coordinates_history) = integrator
            .integrate(
                |_: Scalar,
                 state_variables: &ViscoplasticStateVariables<G>,
                 nodal_coordinates: &NodalCoordinatesBlock| {
                    Ok(self.state_variables_evolution(nodal_coordinates, state_variables)?)
                },
                |t: Scalar,
                 state_variables: &ViscoplasticStateVariables<G>,
                 nodal_coordinates: &NodalCoordinatesBlock| {
                    Ok(self.root_inner(
                        EqualityConstraint::Linear(foo.0.clone(), foo.1(t)),
                        state_variables,
                        &solver,
                        nodal_coordinates,
                    )?)
                },
                time,
                self.elements()
                    .iter()
                    .map(|element| {
                        from_fn(|_| {
                            (
                                DeformationGradientPlastic::identity(),
                                *element.constitutive_model().initial_yield_stress(),
                            )
                                .into()
                        })
                        .into()
                    })
                    .collect(),
                self.coordinates().clone().into(),
            )?;
        Ok((
            time_history,
            nodal_coordinates_history,
            state_variables_history,
        ))
    }
    fn root_inner(
        &self,
        equality_constraint: EqualityConstraint,
        state_variables: &ViscoplasticStateVariables<G>,
        solver: &impl FirstOrderRootFinding<
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
        initial_guess: &NodalCoordinatesBlock,
    ) -> Result<NodalCoordinatesBlock, OptimizationError> {
        solver.root(
            |nodal_coordinates: &NodalCoordinatesBlock| {
                Ok(self.nodal_forces(nodal_coordinates, state_variables)?)
            },
            |nodal_coordinates: &NodalCoordinatesBlock| {
                Ok(self.nodal_stiffnesses(nodal_coordinates, state_variables)?)
            },
            initial_guess.clone(),
            equality_constraint.clone(),
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
    ) -> Result<NodalForcesBlock, FiniteElementBlockError> {
        let mut nodal_forces = NodalForcesBlock::zero(nodal_coordinates.len());
        match self
            .elements()
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
                Ok::<(), FiniteElementError>(())
            }) {
            Ok(()) => Ok(nodal_forces),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        nodal_velocities: &NodalVelocitiesBlock,
    ) -> Result<NodalStiffnessesBlock, FiniteElementBlockError> {
        let mut nodal_stiffnesses = NodalStiffnessesBlock::zero(nodal_coordinates.len());
        match self
            .elements()
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
                Ok::<(), FiniteElementError>(())
            }) {
            Ok(()) => Ok(nodal_stiffnesses),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
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
                solution =
                    self.root_inner(&equality_constraint, nodal_coordinates, &solver, &solution)?;
                Ok(solution.clone())
            },
            time,
            self.coordinates().clone().into(),
        )
    }
    fn root_inner(
        &self,
        equality_constraint: &EqualityConstraint,
        nodal_coordinates: &NodalCoordinatesBlock,
        solver: &impl FirstOrderRootFinding<
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
        initial_guess: &NodalVelocitiesBlock,
    ) -> Result<NodalVelocitiesBlock, OptimizationError> {
        solver.root(
            |nodal_velocities: &NodalVelocitiesBlock| {
                Ok(self.nodal_forces(nodal_coordinates, nodal_velocities)?)
            },
            |nodal_velocities: &NodalVelocitiesBlock| {
                Ok(self.nodal_stiffnesses(nodal_coordinates, nodal_velocities)?)
            },
            initial_guess.clone(),
            equality_constraint.clone(),
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
    ) -> Result<Scalar, FiniteElementBlockError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity().iter())
            .map(|(element, element_connectivity)| {
                element.viscous_dissipation(
                    &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                    &self.nodal_velocities_element(element_connectivity, nodal_velocities),
                )
            })
            .sum()
        {
            Ok(viscous_dissipation) => Ok(viscous_dissipation),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
        nodal_velocities: &NodalVelocitiesBlock,
    ) -> Result<Scalar, FiniteElementBlockError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity().iter())
            .map(|(element, element_connectivity)| {
                element.dissipation_potential(
                    &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                    &self.nodal_velocities_element(element_connectivity, nodal_velocities),
                )
            })
            .sum()
        {
            Ok(dissipation_potential) => Ok(dissipation_potential),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
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
                    &equality_constraint,
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
    fn minimize_inner(
        &self,
        equality_constraint: &EqualityConstraint,
        nodal_coordinates: &NodalCoordinatesBlock,
        solver: &impl SecondOrderOptimization<
            Scalar,
            NodalForcesBlock,
            NodalStiffnessesBlock,
            NodalCoordinatesBlock,
        >,
        initial_guess: &NodalVelocitiesBlock,
    ) -> Result<NodalVelocitiesBlock, OptimizationError> {
        let num_coords = nodal_coordinates.len();
        let banded = band(self.connectivity(), equality_constraint, num_coords);
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
            equality_constraint.clone(),
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
    ) -> Result<Scalar, FiniteElementBlockError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity().iter())
            .map(|(element, element_connectivity)| {
                element.helmholtz_free_energy(
                    &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                )
            })
            .sum()
        {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
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
