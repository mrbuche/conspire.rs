use crate::{
    fem::{
        Blocks, FiniteElementModel, FiniteElementModelError, FiniteElements, FirstOrderMinimize,
        FirstOrderRoot, Model, SecondOrderMinimize, ZerothOrderRoot,
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

pub trait ThermalConductionFiniteElements
where
    Self: FiniteElements,
{
    fn potential(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Scalar, FiniteElementModelError>;
    fn nodal_forces(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalForcesThermal, FiniteElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalStiffnessesThermal, FiniteElementModelError>;
}

impl<B> ThermalConductionFiniteElements for Model<B>
where
    B: ThermalConductionFiniteElements,
{
    fn potential(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Scalar, FiniteElementModelError> {
        self.blocks.potential(nodal_temperatures)
    }
    fn nodal_forces(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalForcesThermal, FiniteElementModelError> {
        self.blocks.nodal_forces(nodal_temperatures)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalStiffnessesThermal, FiniteElementModelError> {
        self.blocks.nodal_stiffnesses(nodal_temperatures)
    }
}

impl<B1, B2> ThermalConductionFiniteElements for Blocks<B1, B2>
where
    B1: ThermalConductionFiniteElements,
    B2: ThermalConductionFiniteElements,
{
    fn potential(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Scalar, FiniteElementModelError> {
        Ok(self.0.potential(nodal_temperatures)? + self.1.potential(nodal_temperatures)?)
    }
    fn nodal_forces(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalForcesThermal, FiniteElementModelError> {
        Ok(self.0.nodal_forces(nodal_temperatures)? + self.1.nodal_forces(nodal_temperatures)?)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalStiffnessesThermal, FiniteElementModelError> {
        Ok(self.0.nodal_stiffnesses(nodal_temperatures)?
            + self.1.nodal_stiffnesses(nodal_temperatures)?)
    }
}

impl<B> ZerothOrderRoot<NodalTemperatures> for Model<B>
where
    B: ThermalConductionFiniteElements,
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

impl<B> FirstOrderRoot<NodalForcesThermal, NodalStiffnessesThermal, NodalTemperatures> for Model<B>
where
    B: ThermalConductionFiniteElements,
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

impl<B> FirstOrderMinimize<Scalar, NodalTemperatures> for Model<B>
where
    B: ThermalConductionFiniteElements,
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

impl<B> SecondOrderMinimize<Scalar, NodalForcesThermal, NodalStiffnessesThermal, NodalTemperatures>
    for Model<B>
where
    B: ThermalConductionFiniteElements,
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
