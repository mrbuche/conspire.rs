#[cfg(test)]
pub mod test;

use crate::{
    constitutive::thermal::conduction::ThermalConduction,
    fem::block::{
        Block, FiniteElementBlockError, FirstOrderMinimize, FirstOrderRoot, SecondOrderMinimize,
        ZerothOrderRoot, band,
        element::{FiniteElementError, thermal::conduction::ThermalConductionFiniteElement},
        thermal::{NodalTemperatures, ThermalFiniteElementBlock},
    },
    math::{
        Scalar, SquareMatrix, Tensor, Vector,
        optimize::{
            EqualityConstraint, FirstOrderOptimization, FirstOrderRootFinding, OptimizationError,
            SecondOrderOptimization, ZerothOrderRootFinding,
        },
    },
};

pub type NodalForcesThermal = Vector;
pub type NodalStiffnessesThermal = SquareMatrix;

pub trait ThermalConductionFiniteElementBlock<
    C,
    F,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    C: ThermalConduction,
    F: ThermalConductionFiniteElement<C, G, M, N, P>,
{
    fn potential(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Scalar, FiniteElementBlockError>;
    fn nodal_forces(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalForcesThermal, FiniteElementBlockError>;
    fn nodal_stiffnesses(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalStiffnessesThermal, FiniteElementBlockError>;
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    ThermalConductionFiniteElementBlock<C, F, G, M, N, P> for Block<C, F, G, M, N, P>
where
    C: ThermalConduction,
    F: ThermalConductionFiniteElement<C, G, M, N, P>,
{
    fn potential(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Scalar, FiniteElementBlockError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, element_connectivity)| {
                element.potential(
                    self.constitutive_model(),
                    &self.nodal_temperatures_element(element_connectivity, nodal_temperatures),
                )
            })
            .sum()
        {
            Ok(potential) => Ok(potential),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_forces(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalForcesThermal, FiniteElementBlockError> {
        let mut nodal_forces = NodalForcesThermal::zero(nodal_temperatures.len());
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, element_connectivity)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &self.nodal_temperatures_element(element_connectivity, nodal_temperatures),
                    )?
                    .iter()
                    .zip(element_connectivity)
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
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalStiffnessesThermal, FiniteElementBlockError> {
        let mut nodal_stiffnesses = NodalStiffnessesThermal::zero(nodal_temperatures.len());
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, element_connectivity)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &self.nodal_temperatures_element(element_connectivity, nodal_temperatures),
                    )?
                    .iter()
                    .zip(element_connectivity)
                    .for_each(|(object, &node_a)| {
                        object.iter().zip(element_connectivity).for_each(
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

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    ZerothOrderRoot<C, F, G, M, N, NodalTemperatures> for Block<C, F, G, M, N, P>
where
    C: ThermalConduction,
    F: ThermalConductionFiniteElement<C, G, M, N, P>,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl ZerothOrderRootFinding<NodalTemperatures>,
    ) -> Result<NodalTemperatures, OptimizationError> {
        solver.root(
            |nodal_temperatures: &NodalTemperatures| Ok(self.nodal_forces(nodal_temperatures)?),
            NodalTemperatures::zero(self.coordinates().len()),
            equality_constraint,
        )
    }
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    FirstOrderRoot<C, F, G, M, N, NodalForcesThermal, NodalStiffnessesThermal, NodalTemperatures>
    for Block<C, F, G, M, N, P>
where
    C: ThermalConduction,
    F: ThermalConductionFiniteElement<C, G, M, N, P>,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderRootFinding<
            NodalForcesThermal,
            NodalStiffnessesThermal,
            NodalTemperatures,
        >,
    ) -> Result<NodalTemperatures, OptimizationError> {
        solver.root(
            |nodal_temperatures: &NodalTemperatures| Ok(self.nodal_forces(nodal_temperatures)?),
            |nodal_temperatures: &NodalTemperatures| {
                Ok(self.nodal_stiffnesses(nodal_temperatures)?)
            },
            NodalTemperatures::zero(self.coordinates().len()),
            equality_constraint,
        )
    }
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    FirstOrderMinimize<C, F, G, M, N, NodalTemperatures> for Block<C, F, G, M, N, P>
where
    C: ThermalConduction,
    F: ThermalConductionFiniteElement<C, G, M, N, P>,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderOptimization<Scalar, NodalTemperatures>,
    ) -> Result<NodalTemperatures, OptimizationError> {
        solver.minimize(
            |nodal_temperatures: &NodalTemperatures| Ok(self.potential(nodal_temperatures)?),
            |nodal_temperatures: &NodalTemperatures| Ok(self.nodal_forces(nodal_temperatures)?),
            NodalTemperatures::zero(self.coordinates().len()),
            equality_constraint,
        )
    }
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    SecondOrderMinimize<
        C,
        F,
        G,
        M,
        N,
        NodalForcesThermal,
        NodalStiffnessesThermal,
        NodalTemperatures,
    > for Block<C, F, G, M, N, P>
where
    C: ThermalConduction,
    F: ThermalConductionFiniteElement<C, G, M, N, P>,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesThermal,
            NodalStiffnessesThermal,
            NodalTemperatures,
        >,
    ) -> Result<NodalTemperatures, OptimizationError> {
        let banded = band(
            self.connectivity(),
            &equality_constraint,
            self.coordinates().len(),
            1,
        );
        solver.minimize(
            |nodal_temperatures: &NodalTemperatures| Ok(self.potential(nodal_temperatures)?),
            |nodal_temperatures: &NodalTemperatures| Ok(self.nodal_forces(nodal_temperatures)?),
            |nodal_temperatures: &NodalTemperatures| {
                Ok(self.nodal_stiffnesses(nodal_temperatures)?)
            },
            NodalTemperatures::zero(self.coordinates().len()),
            equality_constraint,
            Some(banded),
        )
    }
}
