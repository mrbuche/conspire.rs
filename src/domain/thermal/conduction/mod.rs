use crate::{
    domain::{
        Blocks, ElementModel, ElementModelError, FirstOrderMinimize, FirstOrderRoot, Model,
        ProvidesTangent, SecondOrderMinimize, SolverFor, ZerothOrderRoot,
        block::{element::Elements, finalize_node_neighbors, solver_from_neighbors},
        thermal::{NodalForcesThermal, NodalStiffnessesThermal, NodalTemperatures},
    },
    math::{
        Quantity, Tensor,
        optimize::{
            EqualityConstraint, NewtonRaphson, Optimization, OptimizationError, RootFinding,
        },
    },
    units::PowerTemperature,
};

pub trait ThermalConductionElements
where
    Self: Elements,
{
    fn potential(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<PowerTemperature>, ElementModelError>;
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
    ) -> Result<Quantity<PowerTemperature>, ElementModelError> {
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
    ) -> Result<Quantity<PowerTemperature>, ElementModelError> {
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

impl<B, const D: usize> ZerothOrderRoot<NodalForcesThermal, NodalTemperatures> for Model<B, D>
where
    B: ThermalConductionElements,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl RootFinding<NodalForcesThermal, (), NodalTemperatures>,
    ) -> Result<NodalTemperatures, OptimizationError> {
        solver.root(
            |nodal_temperatures: &NodalTemperatures| Ok(self.nodal_forces(nodal_temperatures)?),
            |_| Ok(()),
            NodalTemperatures::zero(self.coordinates().len()),
            equality_constraint,
            None,
        )
    }
}

impl<B, const D: usize> SolverFor<Model<B, D>, NodalForcesThermal, NodalStiffnessesThermal>
    for NewtonRaphson
where
    B: ThermalConductionElements,
{
    type Tangent = NodalStiffnessesThermal;
    const SPARSE: bool = true;
}

impl<B, const D: usize>
    FirstOrderRoot<NodalForcesThermal, NodalStiffnessesThermal, NodalTemperatures> for Model<B, D>
where
    B: ThermalConductionElements,
{
    fn root<S>(
        &self,
        equality_constraint: EqualityConstraint,
        solver: S,
    ) -> Result<NodalTemperatures, OptimizationError>
    where
        S: SolverFor<Self, NodalForcesThermal, NodalStiffnessesThermal>
            + RootFinding<NodalForcesThermal, S::Tangent, NodalTemperatures>,
        Self: ProvidesTangent<NodalTemperatures, S::Tangent>,
    {
        let sparse = S::SPARSE.then(|| {
            let mut neighbors = vec![Vec::new(); self.coordinates().len()];
            self.node_neighbors(&mut neighbors);
            finalize_node_neighbors(&mut neighbors);
            solver_from_neighbors(&neighbors, &equality_constraint, 1, true)
        });
        solver.root(
            |nodal_temperatures: &NodalTemperatures| Ok(self.nodal_forces(nodal_temperatures)?),
            |nodal_temperatures: &NodalTemperatures| Ok(self.provide_tangent(nodal_temperatures)?),
            NodalTemperatures::zero(self.coordinates().len()),
            equality_constraint,
            sparse,
        )
    }
}

impl<B, const D: usize>
    FirstOrderMinimize<Quantity<PowerTemperature>, NodalForcesThermal, NodalTemperatures>
    for Model<B, D>
where
    B: ThermalConductionElements,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl Optimization<Quantity<PowerTemperature>, NodalForcesThermal, (), NodalTemperatures>,
    ) -> Result<NodalTemperatures, OptimizationError> {
        solver.minimize(
            |nodal_temperatures: &NodalTemperatures| Ok(self.potential(nodal_temperatures)?),
            |nodal_temperatures: &NodalTemperatures| Ok(self.nodal_forces(nodal_temperatures)?),
            |_| Ok(()),
            NodalTemperatures::zero(self.coordinates().len()),
            equality_constraint,
            None,
        )
    }
}

impl<B, const D: usize> SolverFor<Model<B, D>, Quantity<PowerTemperature>, NodalForcesThermal>
    for NewtonRaphson
where
    B: ThermalConductionElements,
{
    type Tangent = NodalStiffnessesThermal;
    const SPARSE: bool = true;
}

impl<B, const D: usize>
    SecondOrderMinimize<Quantity<PowerTemperature>, NodalForcesThermal, NodalTemperatures>
    for Model<B, D>
where
    B: ThermalConductionElements,
{
    fn minimize<S>(
        &self,
        equality_constraint: EqualityConstraint,
        solver: S,
    ) -> Result<NodalTemperatures, OptimizationError>
    where
        S: SolverFor<Self, Quantity<PowerTemperature>, NodalForcesThermal>
            + Optimization<
                Quantity<PowerTemperature>,
                NodalForcesThermal,
                S::Tangent,
                NodalTemperatures,
            >,
        Self: ProvidesTangent<NodalTemperatures, S::Tangent>,
    {
        let sparse = S::SPARSE.then(|| {
            let mut neighbors = vec![Vec::new(); self.coordinates().len()];
            self.node_neighbors(&mut neighbors);
            finalize_node_neighbors(&mut neighbors);
            solver_from_neighbors(&neighbors, &equality_constraint, 1, true)
        });
        solver.minimize(
            |nodal_temperatures: &NodalTemperatures| Ok(self.potential(nodal_temperatures)?),
            |nodal_temperatures: &NodalTemperatures| Ok(self.nodal_forces(nodal_temperatures)?),
            |nodal_temperatures: &NodalTemperatures| Ok(self.provide_tangent(nodal_temperatures)?),
            NodalTemperatures::zero(self.coordinates().len()),
            equality_constraint,
            sparse,
        )
    }
}
