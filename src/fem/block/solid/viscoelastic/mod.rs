use crate::{
    constitutive::solid::viscoelastic::Viscoelastic,
    fem::{
        NodalCoordinatesBlock, NodalCoordinatesHistory, NodalForcesBlock, NodalStiffnessesBlock,
        NodalVelocities, NodalVelocitiesBlock, NodalVelocitiesHistory,
        block::{
            ElementBlock, FiniteElementBlockError,
            element::{FiniteElementError, ViscoelasticFiniteElement},
            solid::SolidFiniteElementBlock,
        },
    },
    math::{
        Scalar, Tensor,
        integrate::{Explicit, IntegrationError},
        optimize::{EqualityConstraint, FirstOrderRootFinding, OptimizationError},
    },
    mechanics::{DeformationGradientRateList, Times},
};

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

impl<C, F, const G: usize, const N: usize> ViscoelasticFiniteElementBlock<C, F, G, N>
    for ElementBlock<C, F, N>
where
    C: Viscoelastic,
    F: ViscoelasticFiniteElement<C, G, N>,
    Self: SolidFiniteElementBlock<C, F, G, N>,
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
                        self.constitutive_model(),
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
                        self.constitutive_model(),
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
            .map(|&node| nodal_velocities[node].clone())
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
            NodalVelocitiesBlock,
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
