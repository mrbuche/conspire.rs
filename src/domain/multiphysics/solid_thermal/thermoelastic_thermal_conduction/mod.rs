use crate::{
    domain::{
        Blocks, ElementModel, ElementModelError, Model, NodalCoordinates, ProvidesTangent, Root,
        SolverFor,
        block::{element::Elements, finalize_node_neighbors, solver_from_neighbors},
        multiphysics::solid_thermal::{
            NodalForceSolidThermal, NodalForcesSolidThermal, NodalStatesSolidThermal,
            NodalStiffnessesSolidThermal,
        },
        thermal::NodalTemperatures,
    },
    math::{
        Tensor, TensorTuple,
        optimize::{EqualityConstraint, NewtonRaphson, OptimizationError, RootFinding},
    },
};

pub trait ThermoelasticThermalConductionElements
where
    Self: Elements,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_forces: &mut NodalForcesSolidThermal,
    ) -> Result<(), ElementModelError>;
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalForcesSolidThermal, ElementModelError> {
        let mut nodal_forces = (0..nodal_coordinates.len())
            .map(|_| NodalForceSolidThermal::default())
            .collect::<NodalForcesSolidThermal>();
        self.nodal_forces_into(nodal_coordinates, nodal_temperatures, &mut nodal_forces)?;
        Ok(nodal_forces)
    }
    fn nodal_reference_temperatures_into(&self, nodal_temperatures: &mut NodalTemperatures);
    fn nodal_reference_temperatures(&self, nodes: usize) -> NodalTemperatures {
        let mut nodal_temperatures = NodalTemperatures::zero(nodes);
        self.nodal_reference_temperatures_into(&mut nodal_temperatures);
        nodal_temperatures
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_stiffnesses: &mut NodalStiffnessesSolidThermal,
    ) -> Result<(), ElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalStiffnessesSolidThermal, ElementModelError> {
        let mut nodal_stiffnesses = NodalStiffnessesSolidThermal::zero(nodal_coordinates.len());
        self.nodal_stiffnesses_into(
            nodal_coordinates,
            nodal_temperatures,
            &mut nodal_stiffnesses,
        )?;
        Ok(nodal_stiffnesses)
    }
}

impl<B> ThermoelasticThermalConductionElements for Model<B, 3>
where
    B: ThermoelasticThermalConductionElements,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_forces: &mut NodalForcesSolidThermal,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_forces_into(nodal_coordinates, nodal_temperatures, nodal_forces)
    }
    fn nodal_reference_temperatures_into(&self, nodal_temperatures: &mut NodalTemperatures) {
        self.blocks
            .nodal_reference_temperatures_into(nodal_temperatures)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_stiffnesses: &mut NodalStiffnessesSolidThermal,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_stiffnesses_into(nodal_coordinates, nodal_temperatures, nodal_stiffnesses)
    }
}

impl<B1, B2> ThermoelasticThermalConductionElements for Blocks<B1, B2>
where
    B1: ThermoelasticThermalConductionElements,
    B2: ThermoelasticThermalConductionElements,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_forces: &mut NodalForcesSolidThermal,
    ) -> Result<(), ElementModelError> {
        self.0
            .nodal_forces_into(nodal_coordinates, nodal_temperatures, nodal_forces)?;
        self.1
            .nodal_forces_into(nodal_coordinates, nodal_temperatures, nodal_forces)
    }
    fn nodal_reference_temperatures_into(&self, nodal_temperatures: &mut NodalTemperatures) {
        self.0.nodal_reference_temperatures_into(nodal_temperatures);
        self.1.nodal_reference_temperatures_into(nodal_temperatures)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_stiffnesses: &mut NodalStiffnessesSolidThermal,
    ) -> Result<(), ElementModelError> {
        self.0
            .nodal_stiffnesses_into(nodal_coordinates, nodal_temperatures, nodal_stiffnesses)?;
        self.1
            .nodal_stiffnesses_into(nodal_coordinates, nodal_temperatures, nodal_stiffnesses)
    }
}

fn split(nodal_states: &NodalStatesSolidThermal) -> (NodalCoordinates<3>, NodalTemperatures) {
    (
        nodal_states.iter().map(|state| state.0.clone()).collect(),
        nodal_states.iter().map(|state| state.1).collect(),
    )
}

impl<B> SolverFor<Model<B, 3>, NodalForcesSolidThermal, NodalStiffnessesSolidThermal>
    for NewtonRaphson
where
    B: ThermoelasticThermalConductionElements,
{
    type Tangent = NodalStiffnessesSolidThermal;
    const SPARSE: bool = true;
}

impl<B> ProvidesTangent<NodalStatesSolidThermal, NodalStiffnessesSolidThermal> for Model<B, 3>
where
    B: ThermoelasticThermalConductionElements,
{
    fn provide_tangent(
        &self,
        nodal_states: &NodalStatesSolidThermal,
    ) -> Result<NodalStiffnessesSolidThermal, ElementModelError> {
        let (nodal_coordinates, nodal_temperatures) = split(nodal_states);
        self.nodal_stiffnesses(&nodal_coordinates, &nodal_temperatures)
    }
}

impl<B> Root<NodalForcesSolidThermal, NodalStiffnessesSolidThermal, NodalStatesSolidThermal>
    for Model<B, 3>
where
    B: ThermoelasticThermalConductionElements,
{
    fn root<S>(
        &self,
        equality_constraint: EqualityConstraint,
        solver: S,
    ) -> Result<NodalStatesSolidThermal, OptimizationError>
    where
        S: SolverFor<Self, NodalForcesSolidThermal, NodalStiffnessesSolidThermal>
            + RootFinding<NodalForcesSolidThermal, S::Tangent, NodalStatesSolidThermal>,
        Self: ProvidesTangent<NodalStatesSolidThermal, S::Tangent>,
    {
        let sparse = S::SPARSE.then(|| {
            let mut neighbors = vec![Vec::new(); self.coordinates().len()];
            self.node_neighbors(&mut neighbors);
            finalize_node_neighbors(&mut neighbors);
            solver_from_neighbors(&neighbors, &equality_constraint, 4, false)
        });
        solver.root(
            |nodal_states: &NodalStatesSolidThermal| {
                let (nodal_coordinates, nodal_temperatures) = split(nodal_states);
                Ok(self.nodal_forces(&nodal_coordinates, &nodal_temperatures)?)
            },
            |nodal_states: &NodalStatesSolidThermal| Ok(self.provide_tangent(nodal_states)?),
            NodalCoordinates::from(self.coordinates().clone())
                .into_iter()
                .zip(self.nodal_reference_temperatures(self.coordinates().len()))
                .map(|(nodal_coordinate, nodal_temperature)| {
                    TensorTuple(nodal_coordinate, nodal_temperature)
                })
                .collect(),
            equality_constraint,
            sparse,
        )
    }
}
