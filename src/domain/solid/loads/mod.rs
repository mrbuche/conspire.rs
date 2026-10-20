use crate::{
    domain::{
        ElementModel, NodalAccelerations, NodalCoordinates, NodalReferenceCoordinates,
        NodalVelocities,
        boundary::{
            Conservative, ConservativeAt, Layout, Load, LoadAt, LoadStiffness, LoadStiffnessAt,
            Loads, PowerAt, Solid, SolidThermal,
        },
        multiphysics::solid_thermal::{NodalForcesSolidThermal, NodalStatesSolidThermal},
        solid::{NodalForcesSolid, mass::InertialMass},
        volume::{NodalVolumeElements, NodalVolumes},
    },
    math::{Current, Quantity, Tensor, TensorRank1},
    mechanics::ReferenceCoordinate,
    units::{Acceleration, Energy, Force, Power, StressPerLength, Time, UnitMul, Volume},
};
use std::marker::PhantomData;

/// A layout whose degrees of freedom begin with the coordinates of a solid in three dimensions.
pub trait SolidLayout: Layout {}

impl SolidLayout for Solid<3> {}

impl SolidLayout for SolidThermal<3> {}

/// Forces on the nodes of a solid that do not change as the body deforms.
#[derive(Clone, Debug)]
pub struct DeadLoad {
    reference: NodalCoordinates<3>,
    forces: NodalForcesSolid<3>,
}

impl DeadLoad {
    pub(crate) fn new(
        coordinates: &NodalReferenceCoordinates<3>,
        forces: NodalForcesSolid<3>,
    ) -> Result<Self, String> {
        if coordinates.len() == forces.len() {
            Ok(Self {
                reference: coordinates.clone().into(),
                forces,
            })
        } else {
            Err("There must be exactly one force for each node.".to_string())
        }
    }
    fn check(&self, nodes: usize) -> Result<(), String> {
        if nodes == self.reference.len() {
            Ok(())
        } else {
            Err("The load is on a mesh with another number of nodes.".to_string())
        }
    }
}

impl<L, T> Loads<L, T>
where
    L: SolidLayout,
{
    /// Adds the weight of a body, which is the force of an acceleration on the masses of its nodes.
    ///
    /// The masses may be lumped or consistent, and the acceleration is the same at every node.
    pub fn body_acceleration(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        masses: &impl InertialMass<3>,
        acceleration: &TensorRank1<3, Current, Acceleration>,
    ) -> Result<Loads<L, (T, DeadLoad)>, String> {
        let accelerations = coordinates
            .iter()
            .map(|_| acceleration.clone())
            .collect::<NodalAccelerations<3>>();
        self.nodal_forces(coordinates, masses.inertial_forces(&accelerations))
    }
    /// Adds a force on each node, such as the weight of a body.
    pub fn nodal_forces(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        forces: NodalForcesSolid<3>,
    ) -> Result<Loads<L, (T, DeadLoad)>, String> {
        Ok(self.with(DeadLoad::new(coordinates, forces)?))
    }
}

impl<X, J> LoadStiffness<X, J> for DeadLoad {}

impl<X, J> LoadStiffnessAt<X, J> for DeadLoad {}

impl<W, G, X, J> LoadStiffnessAt<X, J> for TimedLoad<W, G> {}

impl Load<NodalCoordinates<3>, NodalForcesSolid<3>> for DeadLoad {
    fn subtract_from(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        residual: &mut NodalForcesSolid<3>,
    ) -> Result<(), String> {
        self.check(nodal_coordinates.len())?;
        residual
            .iter_mut()
            .zip(&self.forces)
            .for_each(|(residual, force)| *residual -= force);
        Ok(())
    }
}

impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>> for DeadLoad {
    fn subtract_at(
        &self,
        _: Quantity<Time>,
        nodal_coordinates: &NodalCoordinates<3>,
        residual: &mut NodalForcesSolid<3>,
    ) -> Result<(), String> {
        self.subtract_from(nodal_coordinates, residual)
    }
}

impl Conservative<NodalCoordinates<3>, Quantity<Energy>> for DeadLoad {
    fn potential_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Energy>, String> {
        self.check(nodal_coordinates.len())?;
        Ok(-nodal_coordinates
            .iter()
            .zip(&self.reference)
            .zip(&self.forces)
            .map(|((coordinate, reference), force)| force * &(coordinate - reference))
            .sum::<Quantity<Energy>>())
    }
}

impl Load<NodalStatesSolidThermal, NodalForcesSolidThermal> for DeadLoad {
    fn subtract_from(
        &self,
        nodal_states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        self.check(nodal_states.len())?;
        residual
            .iter_mut()
            .zip(&self.forces)
            .for_each(|(residual, force)| residual.0 -= force);
        Ok(())
    }
}

impl LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal> for DeadLoad {
    fn subtract_at(
        &self,
        _: Quantity<Time>,
        nodal_states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        self.subtract_from(nodal_states, residual)
    }
}

impl ConservativeAt<NodalCoordinates<3>, Quantity<Energy>> for DeadLoad {
    fn potential_energy_at(
        &self,
        _: Quantity<Time>,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Energy>, String> {
        self.potential_energy(nodal_coordinates)
    }
}

impl PowerAt<NodalCoordinates<3>, NodalVelocities<3>, Quantity<Power>> for DeadLoad {
    fn power_at(
        &self,
        _: Quantity<Time>,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Result<Quantity<Power>, String> {
        self.check(nodal_coordinates.len())?;
        Ok(-nodal_velocities
            .iter()
            .zip(&self.forces)
            .map(|(velocity, force)| force * velocity)
            .sum::<Quantity<Power>>())
    }
}

/// A force per unit volume, such as a weight.
pub type BodyForce = TensorRank1<3, Current, StressPerLength>;

type WithBodyForce<L, T> = Result<Loads<L, (T, DeadLoad)>, String>;
type WithTimedBodyForce<L, T, G> = Result<Loads<L, (T, TimedLoad<Volume, G>)>, String>;

impl<L, T> Loads<L, T>
where
    L: SolidLayout,
{
    /// Adds a force per unit reference volume, the same at every node.
    pub fn body_force(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        volumes: &NodalVolumes,
        force: &BodyForce,
    ) -> WithBodyForce<L, T> {
        self.nodal_forces(
            coordinates,
            volumes.iter().map(|&volume| force * volume).collect(),
        )
    }
    /// Adds a force per unit reference volume that is a function of position, which is
    /// integrated against the shape functions of the nodes at the samples of the elements.
    pub fn body_force_field(
        self,
        model: &(impl NodalVolumeElements + ElementModel<3>),
        field: impl Fn(&ReferenceCoordinate) -> BodyForce,
    ) -> WithBodyForce<L, T> {
        let coordinates = model.coordinates();
        let mut forces = NodalForcesSolid::<3>::zero(coordinates.len());
        model.for_each_sample(coordinates, &mut |node, position, volume| {
            forces[node] += &field(position) * volume
        });
        self.nodal_forces(coordinates, forces)
    }
    /// Adds a force per unit reference volume that is a function of time.
    pub fn body_force_with<G>(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        volumes: &NodalVolumes,
        force: G,
    ) -> WithTimedBodyForce<L, T, G>
    where
        G: Fn(Quantity<Time>) -> BodyForce,
    {
        Ok(self.with(TimedLoad::new(
            coordinates,
            volumes.iter().copied().collect(),
            force,
        )?))
    }
}

/// A force per unit of some measure, such as area or volume, on the nodes of a solid, whose
/// force does not change as the body deforms, and is a function of time.
pub struct TimedLoad<W, G> {
    weights: Vec<Quantity<W>>,
    reference: NodalCoordinates<3>,
    density: G,
    unit: PhantomData<W>,
}

impl<W, G> TimedLoad<W, G> {
    pub(crate) fn new(
        coordinates: &NodalReferenceCoordinates<3>,
        weights: Vec<Quantity<W>>,
        density: G,
    ) -> Result<Self, String> {
        if weights.len() == coordinates.len() {
            Ok(Self {
                weights,
                reference: coordinates.clone().into(),
                density,
                unit: PhantomData,
            })
        } else {
            Err("There must be exactly one weight for each node.".to_string())
        }
    }
}

impl<W, D, G> TimedLoad<W, G>
where
    G: Fn(Quantity<Time>) -> TensorRank1<3, Current, D>,
    D: UnitMul<W, Output = Force>,
{
    fn forces(&self, time: Quantity<Time>, nodes: usize) -> Result<NodalForcesSolid<3>, String> {
        if nodes != self.weights.len() {
            return Err("The load is on a mesh with another number of nodes.".to_string());
        }
        let density = (self.density)(time);
        Ok(self
            .weights
            .iter()
            .map(|&weight| &density * weight)
            .collect())
    }
}

impl<W, D, G> LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>> for TimedLoad<W, G>
where
    G: Fn(Quantity<Time>) -> TensorRank1<3, Current, D>,
    D: UnitMul<W, Output = Force>,
{
    fn subtract_at(
        &self,
        time: Quantity<Time>,
        nodal_coordinates: &NodalCoordinates<3>,
        residual: &mut NodalForcesSolid<3>,
    ) -> Result<(), String> {
        residual
            .iter_mut()
            .zip(self.forces(time, nodal_coordinates.len())?)
            .for_each(|(residual, force)| *residual -= force);
        Ok(())
    }
}

impl<W, D, G> ConservativeAt<NodalCoordinates<3>, Quantity<Energy>> for TimedLoad<W, G>
where
    G: Fn(Quantity<Time>) -> TensorRank1<3, Current, D>,
    D: UnitMul<W, Output = Force>,
{
    fn potential_energy_at(
        &self,
        time: Quantity<Time>,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Energy>, String> {
        Ok(-nodal_coordinates
            .iter()
            .zip(&self.reference)
            .zip(self.forces(time, nodal_coordinates.len())?)
            .map(|((coordinate, reference), force)| force * &(coordinate - reference))
            .sum::<Quantity<Energy>>())
    }
}

impl<W, D, G> PowerAt<NodalCoordinates<3>, NodalVelocities<3>, Quantity<Power>> for TimedLoad<W, G>
where
    G: Fn(Quantity<Time>) -> TensorRank1<3, Current, D>,
    D: UnitMul<W, Output = Force>,
{
    fn power_at(
        &self,
        time: Quantity<Time>,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Result<Quantity<Power>, String> {
        Ok(-nodal_velocities
            .iter()
            .zip(self.forces(time, nodal_coordinates.len())?)
            .map(|(velocity, force)| force * velocity)
            .sum::<Quantity<Power>>())
    }
}

impl<W, D, G> LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal> for TimedLoad<W, G>
where
    G: Fn(Quantity<Time>) -> TensorRank1<3, Current, D>,
    D: UnitMul<W, Output = Force>,
{
    fn subtract_at(
        &self,
        time: Quantity<Time>,
        nodal_states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        residual
            .iter_mut()
            .zip(self.forces(time, nodal_states.len())?)
            .for_each(|(residual, force)| residual.0 -= force);
        Ok(())
    }
}
