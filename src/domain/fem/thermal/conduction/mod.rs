use crate::{
    fem::{
        Blocks, ElementModel, ElementModelError, Elements, FirstOrderMinimize, FirstOrderRoot,
        Model, SecondOrderMinimize, ZerothOrderRoot,
        block::{
            finalize_node_neighbors, solver_from_neighbors,
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
    fn nodal_forces_into(
        &self,
        nodal_temperatures: &NodalTemperatures,
        nodal_forces: &mut NodalForcesThermal,
    ) -> Result<(), ElementModelError>;
    fn nodal_forces(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalForcesThermal, ElementModelError> {
        let mut nodal_forces = NodalForcesThermal::zero(nodal_temperatures.len());
        self.nodal_forces_into(nodal_temperatures, &mut nodal_forces)?;
        Ok(nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_temperatures: &NodalTemperatures,
        nodal_stiffnesses: &mut NodalStiffnessesThermal,
    ) -> Result<(), ElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalStiffnessesThermal, ElementModelError> {
        let mut nodal_stiffnesses = NodalStiffnessesThermal::zero(nodal_temperatures.len());
        self.nodal_stiffnesses_into(nodal_temperatures, &mut nodal_stiffnesses)?;
        Ok(nodal_stiffnesses)
    }
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
    fn nodal_forces_into(
        &self,
        nodal_temperatures: &NodalTemperatures,
        nodal_forces: &mut NodalForcesThermal,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_forces_into(nodal_temperatures, nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_temperatures: &NodalTemperatures,
        nodal_stiffnesses: &mut NodalStiffnessesThermal,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_stiffnesses_into(nodal_temperatures, nodal_stiffnesses)
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
    fn nodal_forces_into(
        &self,
        nodal_temperatures: &NodalTemperatures,
        nodal_forces: &mut NodalForcesThermal,
    ) -> Result<(), ElementModelError> {
        self.0.nodal_forces_into(nodal_temperatures, nodal_forces)?;
        self.1.nodal_forces_into(nodal_temperatures, nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_temperatures: &NodalTemperatures,
        nodal_stiffnesses: &mut NodalStiffnessesThermal,
    ) -> Result<(), ElementModelError> {
        self.0
            .nodal_stiffnesses_into(nodal_temperatures, nodal_stiffnesses)?;
        self.1
            .nodal_stiffnesses_into(nodal_temperatures, nodal_stiffnesses)
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
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let sparse = solver_from_neighbors(&neighbors, &equality_constraint, 1, true);
        solver.root(
            |nodal_temperatures: &NodalTemperatures| Ok(self.nodal_forces(nodal_temperatures)?),
            |nodal_temperatures: &NodalTemperatures| {
                Ok(self.nodal_stiffnesses(nodal_temperatures)?)
            },
            NodalTemperatures::zero(self.coordinates().len()),
            equality_constraint,
            Some(sparse),
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
        let sparse = solver_from_neighbors(&neighbors, &equality_constraint, 1, true);
        solver.minimize(
            |nodal_temperatures: &NodalTemperatures| Ok(self.potential(nodal_temperatures)?),
            |nodal_temperatures: &NodalTemperatures| Ok(self.nodal_forces(nodal_temperatures)?),
            |nodal_temperatures: &NodalTemperatures| {
                Ok(self.nodal_stiffnesses(nodal_temperatures)?)
            },
            NodalTemperatures::zero(self.coordinates().len()),
            equality_constraint,
            Some(sparse),
        )
    }
}
