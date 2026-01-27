use crate::{
    constitutive::solid::elastic_viscoplastic::ElasticViscoplastic,
    fem::{
        NodalCoordinates, NodalCoordinatesHistory,
        block::{
            Block, FiniteElementBlockError,
            element::{
                FiniteElementError, solid::elastic_viscoplastic::ElasticViscoplasticFiniteElement,
            },
            solid::{NodalForcesSolid, NodalStiffnessesSolid, SolidFiniteElementBlock},
        },
    },
    math::{
        Scalar, Tensor, TensorArray, TensorTupleListVec, TensorTupleListVec2D,
        integrate::{ExplicitInternalVariables, IntegrationError},
        optimize::{EqualityConstraint, FirstOrderRootFinding, OptimizationError},
    },
    mechanics::{DeformationGradientPlastic, Times},
};
use std::array::from_fn;

pub type ViscoplasticStateVariables<const G: usize> =
    TensorTupleListVec<DeformationGradientPlastic, Scalar, G>;

pub type ViscoplasticStateVariablesHistory<const G: usize> =
    TensorTupleListVec2D<DeformationGradientPlastic, Scalar, G>;

pub type ElasticViscoplasticBCs = (crate::math::Matrix, fn(Scalar) -> crate::math::Vector);

pub trait ElasticViscoplasticFiniteElementBlock<
    C,
    F,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    C: ElasticViscoplastic,
    F: ElasticViscoplasticFiniteElement<C, G, M, N, P>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<NodalForcesSolid, FiniteElementBlockError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<NodalStiffnessesSolid, FiniteElementBlockError>;
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<ViscoplasticStateVariables<G>, FiniteElementBlockError>;
    fn root(
        &self,
        bcs: ElasticViscoplasticBCs,
        integrator: impl ExplicitInternalVariables<
            ViscoplasticStateVariables<G>,
            NodalCoordinates,
            ViscoplasticStateVariablesHistory<G>,
            NodalCoordinatesHistory,
        >,
        time: &[Scalar],
        solver: impl FirstOrderRootFinding<NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>,
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
        solver: &impl FirstOrderRootFinding<NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>,
        initial_guess: &NodalCoordinates,
    ) -> Result<NodalCoordinates, OptimizationError>;
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    ElasticViscoplasticFiniteElementBlock<C, F, G, M, N, P> for Block<C, F, G, M, N, P>
where
    C: ElasticViscoplastic,
    F: ElasticViscoplasticFiniteElement<C, G, M, N, P>,
    Self: SolidFiniteElementBlock<C, F, G, M, N, P>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<NodalForcesSolid, FiniteElementBlockError> {
        let mut nodal_forces = NodalForcesSolid::zero(nodal_coordinates.len());
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .zip(state_variables)
            .try_for_each(|((element, nodes), state_variables_element)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                        state_variables_element,
                    )?
                    .iter()
                    .zip(nodes)
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
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<NodalStiffnessesSolid, FiniteElementBlockError> {
        let mut nodal_stiffnesses = NodalStiffnessesSolid::zero(nodal_coordinates.len());
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .zip(state_variables)
            .try_for_each(|((element, nodes), state_variables_element)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                        state_variables_element,
                    )?
                    .iter()
                    .zip(nodes)
                    .for_each(|(object, &node_a)| {
                        object
                            .iter()
                            .zip(nodes)
                            .for_each(|(nodal_stiffness, &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                            })
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
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<ViscoplasticStateVariables<G>, FiniteElementBlockError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .zip(state_variables)
            .map(|((element, nodes), element_state_variables)| {
                element.state_variables_evolution(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    element_state_variables,
                )
            })
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
        bcs: ElasticViscoplasticBCs,
        integrator: impl ExplicitInternalVariables<
            ViscoplasticStateVariables<G>,
            NodalCoordinates,
            ViscoplasticStateVariablesHistory<G>,
            NodalCoordinatesHistory,
        >,
        time: &[Scalar],
        solver: impl FirstOrderRootFinding<NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>,
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
                 nodal_coordinates: &NodalCoordinates| {
                    Ok(self.state_variables_evolution(nodal_coordinates, state_variables)?)
                },
                |t: Scalar,
                 state_variables: &ViscoplasticStateVariables<G>,
                 nodal_coordinates: &NodalCoordinates| {
                    Ok(self.root_inner(
                        EqualityConstraint::Linear(bcs.0.clone(), bcs.1(t)),
                        state_variables,
                        &solver,
                        nodal_coordinates,
                    )?)
                },
                time,
                self.elements()
                    .iter()
                    .map(|_| {
                        from_fn(|_| {
                            (
                                DeformationGradientPlastic::identity(),
                                self.constitutive_model().initial_yield_stress(),
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
        solver: &impl FirstOrderRootFinding<NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>,
        initial_guess: &NodalCoordinates,
    ) -> Result<NodalCoordinates, OptimizationError> {
        solver.root(
            |nodal_coordinates: &NodalCoordinates| {
                Ok(self.nodal_forces(nodal_coordinates, state_variables)?)
            },
            |nodal_coordinates: &NodalCoordinates| {
                Ok(self.nodal_stiffnesses(nodal_coordinates, state_variables)?)
            },
            initial_guess.clone(),
            equality_constraint.clone(),
        )
    }
}
