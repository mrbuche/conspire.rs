use crate::{
    fem::{
        Blocks, ElementModel, ElementModelError, Elements, FirstOrderMinimize, FirstOrderRoot,
        Model, SecondOrderMinimize, ZerothOrderRoot,
        block::{
            band_from_neighbors, finalize_node_neighbors,
            thermal::{
                NodalTemperatures,
                conduction::{NodalForcesThermal, NodalStiffnessesThermal},
            },
        },
    },
    math::{
        Scalar, Tensor,
        optimize::{
            EqualityConstraint, FirstOrderOptimization, FirstOrderRootFinding, OptimizationError,
            SecondOrderOptimization, ZerothOrderRootFinding,
        },
    },
};

pub trait ThermalConductionElements
where
    Self: Elements,
{
    fn potential(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Scalar, ElementModelError>;
    fn nodal_forces(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalForcesThermal, ElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalStiffnessesThermal, ElementModelError>;
}

impl<B, const D: usize> ThermalConductionElements for Model<B, D>
where
    B: ThermalConductionElements,
{
    fn potential(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Scalar, ElementModelError> {
        self.blocks.potential(nodal_temperatures)
    }
    fn nodal_forces(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalForcesThermal, ElementModelError> {
        self.blocks.nodal_forces(nodal_temperatures)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalStiffnessesThermal, ElementModelError> {
        self.blocks.nodal_stiffnesses(nodal_temperatures)
    }
}

impl<B1, B2> ThermalConductionElements for Blocks<B1, B2>
where
    B1: ThermalConductionElements,
    B2: ThermalConductionElements,
{
    fn potential(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Scalar, ElementModelError> {
        Ok(self.0.potential(nodal_temperatures)? + self.1.potential(nodal_temperatures)?)
    }
    fn nodal_forces(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalForcesThermal, ElementModelError> {
        Ok(self.0.nodal_forces(nodal_temperatures)? + self.1.nodal_forces(nodal_temperatures)?)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalStiffnessesThermal, ElementModelError> {
        Ok(self.0.nodal_stiffnesses(nodal_temperatures)?
            + self.1.nodal_stiffnesses(nodal_temperatures)?)
    }
}

impl<B, const D: usize> ZerothOrderRoot<NodalTemperatures> for Model<B, D>
where
    B: ThermalConductionElements,
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

impl<B, const D: usize>
    FirstOrderRoot<NodalForcesThermal, NodalStiffnessesThermal, NodalTemperatures> for Model<B, D>
where
    B: ThermalConductionElements,
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

impl<B, const D: usize> FirstOrderMinimize<Scalar, NodalTemperatures> for Model<B, D>
where
    B: ThermalConductionElements,
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

impl<B, const D: usize>
    SecondOrderMinimize<Scalar, NodalForcesThermal, NodalStiffnessesThermal, NodalTemperatures>
    for Model<B, D>
where
    B: ThermalConductionElements,
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
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let banded = band_from_neighbors(&neighbors, &equality_constraint, 1);
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
