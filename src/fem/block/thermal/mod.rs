use crate::{
    constitutive::thermal::conduction::ThermalConduction,
    fem::{
        NodalForcesBlockThermal, NodalStiffnessesBlockThermal, NodalTemperatures,
        NodalTemperaturesBlock,
        block::{
            ElementBlock, FiniteElementBlockError, FirstOrderRoot, ZerothOrderRoot,
            element::{FiniteElementError, ThermalConductionFiniteElement, ThermalFiniteElement},
        },
    },
    math::{
        Tensor,
        optimize::{
            EqualityConstraint, FirstOrderRootFinding, OptimizationError, ZerothOrderRootFinding,
        },
    },
    mechanics::TemperatureGradients,
};

pub trait ThermalFiniteElementBlock<C, F, const G: usize, const N: usize>
where
    F: ThermalFiniteElement<G, N>,
{
    fn nodal_temperatures_element(
        &self,
        element_connectivity: &[usize; N],
        nodal_temperatures: &NodalTemperaturesBlock,
    ) -> NodalTemperatures<N>;
    fn temperature_gradients(
        &self,
        nodal_temperatures: &NodalTemperaturesBlock,
    ) -> Vec<TemperatureGradients<G>>;
}

impl<C, F, const G: usize, const N: usize> ThermalFiniteElementBlock<C, F, G, N>
    for ElementBlock<C, F, N>
where
    F: ThermalFiniteElement<G, N>,
{
    fn nodal_temperatures_element(
        &self,
        element_connectivity: &[usize; N],
        nodal_temperatures: &NodalTemperaturesBlock,
    ) -> NodalTemperatures<N> {
        element_connectivity
            .iter()
            .map(|&node| nodal_temperatures[node])
            .collect()
    }
    fn temperature_gradients(
        &self,
        nodal_temperatures: &NodalTemperaturesBlock,
    ) -> Vec<TemperatureGradients<G>> {
        self.elements()
            .iter()
            .zip(self.connectivity().iter())
            .map(|(element, element_connectivity)| {
                element.temperature_gradients(
                    &self.nodal_temperatures_element(element_connectivity, nodal_temperatures),
                )
            })
            .collect()
    }
}

pub trait ThermalConductionFiniteElementBlock<C, F, const G: usize, const N: usize>
where
    C: ThermalConduction,
    F: ThermalConductionFiniteElement<C, G, N>,
{
    fn nodal_forces(
        &self,
        nodal_temperatures: &NodalTemperaturesBlock,
    ) -> Result<NodalForcesBlockThermal, FiniteElementBlockError>;
    fn nodal_stiffnesses(
        &self,
        nodal_temperatures: &NodalTemperaturesBlock,
    ) -> Result<NodalStiffnessesBlockThermal, FiniteElementBlockError>;
}

impl<C, F, const G: usize, const N: usize> ThermalConductionFiniteElementBlock<C, F, G, N>
    for ElementBlock<C, F, N>
where
    C: ThermalConduction,
    F: ThermalConductionFiniteElement<C, G, N>,
{
    fn nodal_forces(
        &self,
        nodal_temperatures: &NodalTemperaturesBlock,
    ) -> Result<NodalForcesBlockThermal, FiniteElementBlockError> {
        let mut nodal_forces = NodalForcesBlockThermal::zero(nodal_temperatures.len());
        match self
            .elements()
            .iter()
            .zip(self.connectivity().iter())
            .try_for_each(|(element, element_connectivity)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &self.nodal_temperatures_element(element_connectivity, nodal_temperatures),
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
        nodal_temperatures: &NodalTemperaturesBlock,
    ) -> Result<NodalStiffnessesBlockThermal, FiniteElementBlockError> {
        let mut nodal_stiffnesses = NodalStiffnessesBlockThermal::zero(nodal_temperatures.len());
        match self
            .elements()
            .iter()
            .zip(self.connectivity().iter())
            .try_for_each(|(element, element_connectivity)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &self.nodal_temperatures_element(element_connectivity, nodal_temperatures),
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

impl<C, F, const G: usize, const N: usize> ZerothOrderRoot<C, F, G, N, NodalTemperaturesBlock>
    for ElementBlock<C, F, N>
where
    C: ThermalConduction,
    F: ThermalConductionFiniteElement<C, G, N>,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl ZerothOrderRootFinding<NodalTemperaturesBlock>,
    ) -> Result<NodalTemperaturesBlock, OptimizationError> {
        solver.root(
            |nodal_temperatures: &NodalTemperaturesBlock| Ok(self.nodal_forces(nodal_temperatures)?),
            todo!("Initial temperature guess?"),
            equality_constraint,
        )
    }
}

impl<C, F, const G: usize, const N: usize>
    FirstOrderRoot<
        C,
        F,
        G,
        N,
        NodalForcesBlockThermal,
        NodalStiffnessesBlockThermal,
        NodalTemperaturesBlock,
    > for ElementBlock<C, F, N>
where
    C: ThermalConduction,
    F: ThermalConductionFiniteElement<C, G, N>,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderRootFinding<
            NodalForcesBlockThermal,
            NodalStiffnessesBlockThermal,
            NodalTemperaturesBlock,
        >,
    ) -> Result<NodalTemperaturesBlock, OptimizationError> {
        solver.root(
            |nodal_temperatures: &NodalTemperaturesBlock| Ok(self.nodal_forces(nodal_temperatures)?),
            |nodal_temperatures: &NodalTemperaturesBlock| {
                Ok(self.nodal_stiffnesses(nodal_temperatures)?)
            },
            todo!("Initial temperature guess?"),
            equality_constraint,
        )
    }
}
