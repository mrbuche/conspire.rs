use crate::{
    constitutive::diffusion::Diffusive,
    domain::{
        ElementModel,
        boundary::{
            Conservative, ConservativeAt, Field, Load, LoadAt, LoadStiffness, LoadStiffnessAt,
            Loads, SolidThermal,
        },
        diffusion::{NodalFields, NodalForces, NodalStiffnesses},
        multiphysics::solid_thermal::{
            NodalForcesSolidThermal, NodalStatesSolidThermal, NodalStiffnessesSolidThermal,
        },
        volume::{NodalVolumeElements, NodalVolumes},
    },
    math::{Quantity, Tensor, TensorTuple},
    mechanics::ReferenceCoordinate,
    units::{Power, PowerDensity, PowerPerTemperature, Temperature, Time, UnitMul, Volume},
};
use std::ops::Mul;

/// Sources that do not change with the field, one on each node, such as heat or charge.
#[derive(Clone, Debug)]
pub struct NodalSource<L>
where
    L: Diffusive,
{
    sources: NodalForces<L>,
}

impl<U, T> Loads<Field<U>, T> {
    /// Adds a source on each node, such as the heating of a body.
    pub fn source<L>(self, sources: NodalForces<L>) -> Loads<Field<U>, (T, NodalSource<L>)>
    where
        L: Diffusive<Field = U>,
    {
        self.with(NodalSource { sources })
    }
}

impl<L> NodalSource<L>
where
    L: Diffusive,
{
    fn check(&self, nodes: usize) -> Result<(), String> {
        if nodes == self.sources.len() {
            Ok(())
        } else {
            Err("The source is on a mesh with another number of nodes.".to_string())
        }
    }
}

impl<L, X, J> LoadStiffness<X, J> for NodalSource<L> where L: Diffusive {}

impl<L, X, J> LoadStiffnessAt<X, J> for NodalSource<L> where L: Diffusive {}

impl<W, G, X, J> LoadStiffnessAt<X, J> for TimedSource<W, G> {}

impl<L> Load<NodalFields<L>, NodalForces<L>> for NodalSource<L>
where
    L: Diffusive,
{
    fn subtract_from(
        &self,
        nodal_fields: &NodalFields<L>,
        residual: &mut NodalForces<L>,
    ) -> Result<(), String> {
        self.check(nodal_fields.len())?;
        residual
            .iter_mut()
            .zip(&self.sources)
            .for_each(|(residual, source)| *residual -= source);
        Ok(())
    }
}

impl<L> LoadAt<NodalFields<L>, NodalForces<L>> for NodalSource<L>
where
    L: Diffusive,
{
    fn subtract_at(
        &self,
        _: Quantity<Time>,
        nodal_fields: &NodalFields<L>,
        residual: &mut NodalForces<L>,
    ) -> Result<(), String> {
        self.subtract_from(nodal_fields, residual)
    }
}

impl<L> Conservative<NodalFields<L>, Quantity<L::Energy>> for NodalSource<L>
where
    L: Diffusive,
    Quantity<L::Field>: Mul<Quantity<L>, Output = Quantity<L::Energy>>,
{
    fn potential_energy(
        &self,
        nodal_fields: &NodalFields<L>,
    ) -> Result<Quantity<L::Energy>, String> {
        self.check(nodal_fields.len())?;
        Ok(-nodal_fields
            .iter()
            .zip(&self.sources)
            .map(|(&field, &source)| field * source)
            .sum::<Quantity<L::Energy>>())
    }
}

impl<L> ConservativeAt<NodalFields<L>, Quantity<L::Energy>> for NodalSource<L>
where
    L: Diffusive,
    Quantity<L::Field>: Mul<Quantity<L>, Output = Quantity<L::Energy>>,
{
    fn potential_energy_at(
        &self,
        _: Quantity<Time>,
        nodal_fields: &NodalFields<L>,
    ) -> Result<Quantity<L::Energy>, String> {
        self.potential_energy(nodal_fields)
    }
}

impl<T> Loads<SolidThermal<3>, T> {
    /// Adds a heat source on each node, such as the heating of a body.
    pub fn source(
        self,
        sources: NodalForces<Power>,
    ) -> Loads<SolidThermal<3>, (T, NodalSource<Power>)> {
        self.with(NodalSource { sources })
    }
}

impl Load<NodalStatesSolidThermal, NodalForcesSolidThermal> for NodalSource<Power> {
    fn subtract_from(
        &self,
        nodal_states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        self.check(nodal_states.len())?;
        residual
            .iter_mut()
            .zip(&self.sources)
            .for_each(|(residual, &source)| residual.1 -= source);
        Ok(())
    }
}

impl LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal> for NodalSource<Power> {
    fn subtract_at(
        &self,
        _: Quantity<Time>,
        nodal_states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        self.subtract_from(nodal_states, residual)
    }
}

/// A source per unit of some measure, such as area or volume, on the nodes, that is a function of
/// time.
pub struct TimedSource<W, G> {
    weights: Vec<Quantity<W>>,
    density: G,
}

impl<W, G> TimedSource<W, G> {
    pub(crate) fn new(weights: Vec<Quantity<W>>, density: G) -> Self {
        Self { weights, density }
    }
}

impl<L, F, W, G> LoadAt<NodalFields<L>, NodalForces<L>> for TimedSource<W, G>
where
    L: Diffusive,
    F: UnitMul<W, Output = L>,
    G: Fn(Quantity<Time>) -> Quantity<F>,
{
    fn subtract_at(
        &self,
        time: Quantity<Time>,
        nodal_fields: &NodalFields<L>,
        residual: &mut NodalForces<L>,
    ) -> Result<(), String> {
        if nodal_fields.len() != self.weights.len() {
            return Err("The source is on a mesh with another number of nodes.".to_string());
        }
        let density = (self.density)(time);
        residual
            .iter_mut()
            .zip(&self.weights)
            .for_each(|(residual, &weight)| *residual -= density * weight);
        Ok(())
    }
}

impl<L, F, W, G> ConservativeAt<NodalFields<L>, Quantity<L::Energy>> for TimedSource<W, G>
where
    L: Diffusive,
    F: UnitMul<W, Output = L>,
    G: Fn(Quantity<Time>) -> Quantity<F>,
    Quantity<L::Field>: Mul<Quantity<L>, Output = Quantity<L::Energy>>,
{
    fn potential_energy_at(
        &self,
        time: Quantity<Time>,
        nodal_fields: &NodalFields<L>,
    ) -> Result<Quantity<L::Energy>, String> {
        if nodal_fields.len() != self.weights.len() {
            return Err("The source is on a mesh with another number of nodes.".to_string());
        }
        let density = (self.density)(time);
        Ok(-nodal_fields
            .iter()
            .zip(&self.weights)
            .map(|(&field, &weight)| field * (density * weight))
            .sum::<Quantity<L::Energy>>())
    }
}

impl<F, W, G> LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal> for TimedSource<W, G>
where
    F: UnitMul<W, Output = Power>,
    G: Fn(Quantity<Time>) -> Quantity<F>,
{
    fn subtract_at(
        &self,
        time: Quantity<Time>,
        nodal_states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        if nodal_states.len() != self.weights.len() {
            return Err("The source is on a mesh with another number of nodes.".to_string());
        }
        let density = (self.density)(time);
        residual
            .iter_mut()
            .zip(&self.weights)
            .for_each(|(residual, &weight)| residual.1 -= density * weight);
        Ok(())
    }
}

/// A source per unit reference volume.
type Density<L> = Quantity<<L as Diffusive>::FluxPerLength>;

type WithSource<U, T, L> = Loads<Field<U>, (T, NodalSource<L>)>;
type WithTimedSource<U, T, G> = Loads<Field<U>, (T, TimedSource<Volume, G>)>;

impl<U, T> Loads<Field<U>, T> {
    /// Adds a source per unit reference volume, the same at every node, such as the heating of a
    /// body.
    pub fn source_density<L>(
        self,
        volumes: &NodalVolumes,
        density: Density<L>,
    ) -> WithSource<U, T, L>
    where
        L: Diffusive<Field = U>,
    {
        self.source(volumes.iter().map(|&volume| density * volume).collect())
    }
    /// Adds a source per unit reference volume that is a function of position, which is
    /// integrated against the shape functions of the nodes at the samples of the elements.
    pub fn source_density_field<L>(
        self,
        model: &(impl NodalVolumeElements + ElementModel<3>),
        field: impl Fn(&ReferenceCoordinate) -> Density<L>,
    ) -> WithSource<U, T, L>
    where
        L: Diffusive<Field = U>,
    {
        let coordinates = model.coordinates();
        let mut sources = NodalForces::<L>::zero(coordinates.len());
        model.for_each_sample(coordinates, &mut |node, position, volume| {
            sources[node] += field(position) * volume
        });
        self.source(sources)
    }
    /// Adds a source per unit reference volume that is a function of time.
    pub fn source_density_with<L, G>(
        self,
        volumes: &NodalVolumes,
        density: G,
    ) -> WithTimedSource<U, T, G>
    where
        L: Diffusive<Field = U>,
        G: Fn(Quantity<Time>) -> Density<L>,
    {
        self.with(TimedSource::new(volumes.iter().copied().collect(), density))
    }
}

type WithHeat<T> = Loads<SolidThermal<3>, (T, NodalSource<Power>)>;
type WithTimedHeat<T, G> = Loads<SolidThermal<3>, (T, TimedSource<Volume, G>)>;

impl<T> Loads<SolidThermal<3>, T> {
    /// Adds a heat source per unit reference volume, the same at every node.
    pub fn source_density(
        self,
        volumes: &NodalVolumes,
        density: Quantity<PowerDensity>,
    ) -> WithHeat<T> {
        self.source(volumes.iter().map(|&volume| density * volume).collect())
    }
    /// Adds a heat source per unit reference volume that is a function of position, which is
    /// integrated against the shape functions of the nodes at the samples of the elements.
    pub fn source_density_field(
        self,
        model: &(impl NodalVolumeElements + ElementModel<3>),
        field: impl Fn(&ReferenceCoordinate) -> Quantity<PowerDensity>,
    ) -> WithHeat<T> {
        let coordinates = model.coordinates();
        let mut sources = NodalForces::<Power>::zero(coordinates.len());
        model.for_each_sample(coordinates, &mut |node, position, volume| {
            sources[node] += field(position) * volume
        });
        self.source(sources)
    }
    /// Adds a heat source per unit reference volume that is a function of time.
    pub fn source_density_with<G>(self, volumes: &NodalVolumes, density: G) -> WithTimedHeat<T, G>
    where
        G: Fn(Quantity<Time>) -> Quantity<PowerDensity>,
    {
        self.with(TimedSource::new(volumes.iter().copied().collect(), density))
    }
}

/// A flux through faces that is proportional to the difference between the field and that of the
/// surroundings, such as the convection of heat, held as the coefficient on each node.
#[derive(Clone, Debug)]
pub struct Convection<L>
where
    L: Diffusive,
{
    ambient: Quantity<L::Field>,
    coefficients: Vec<Quantity<L::Stiffness>>,
}

impl<U, T> Loads<Field<U>, T> {
    /// Adds a flux that is the coefficient of each node times the difference of its field from
    /// the ambient field.
    pub fn convection<L>(
        self,
        ambient: Quantity<L::Field>,
        coefficients: Vec<Quantity<L::Stiffness>>,
    ) -> Loads<Field<U>, (T, Convection<L>)>
    where
        L: Diffusive<Field = U>,
    {
        self.with(Convection {
            ambient,
            coefficients,
        })
    }
}

impl<T> Loads<SolidThermal<3>, T> {
    /// Adds a heat flux that is the coefficient of each node times the difference of its
    /// temperature from the ambient temperature.
    pub fn convection(
        self,
        ambient: Quantity<Temperature>,
        coefficients: Vec<Quantity<PowerPerTemperature>>,
    ) -> Loads<SolidThermal<3>, (T, Convection<Power>)> {
        self.with(Convection {
            ambient,
            coefficients,
        })
    }
}

impl<L> Convection<L>
where
    L: Diffusive,
{
    fn check(&self, nodes: usize) -> Result<(), String> {
        if nodes == self.coefficients.len() {
            Ok(())
        } else {
            Err("The convection is on a mesh with another number of nodes.".to_string())
        }
    }
}

impl<L> Load<NodalFields<L>, NodalForces<L>> for Convection<L>
where
    L: Diffusive,
    Quantity<L::Stiffness>: Mul<Quantity<L::Field>, Output = Quantity<L>>,
{
    fn subtract_from(
        &self,
        nodal_fields: &NodalFields<L>,
        residual: &mut NodalForces<L>,
    ) -> Result<(), String> {
        self.check(nodal_fields.len())?;
        residual
            .iter_mut()
            .zip(nodal_fields.iter().zip(&self.coefficients))
            .for_each(|(residual, (&field, &coefficient))| {
                *residual += coefficient * (field - self.ambient)
            });
        Ok(())
    }
}

impl<L> LoadStiffness<NodalFields<L>, NodalStiffnesses<L>> for Convection<L>
where
    L: Diffusive,
{
    fn add_stiffness_to(
        &self,
        nodal_fields: &NodalFields<L>,
        tangent: &mut NodalStiffnesses<L>,
    ) -> Result<(), String> {
        self.check(nodal_fields.len())?;
        self.coefficients
            .iter()
            .enumerate()
            .for_each(|(node, &coefficient)| tangent[node][node] += coefficient);
        Ok(())
    }
}

impl<L> LoadAt<NodalFields<L>, NodalForces<L>> for Convection<L>
where
    L: Diffusive,
    Quantity<L::Stiffness>: Mul<Quantity<L::Field>, Output = Quantity<L>>,
{
    fn subtract_at(
        &self,
        _: Quantity<Time>,
        nodal_fields: &NodalFields<L>,
        residual: &mut NodalForces<L>,
    ) -> Result<(), String> {
        self.subtract_from(nodal_fields, residual)
    }
}

impl<L> LoadStiffnessAt<NodalFields<L>, NodalStiffnesses<L>> for Convection<L>
where
    L: Diffusive,
{
    fn add_stiffness_at(
        &self,
        _: Quantity<Time>,
        nodal_fields: &NodalFields<L>,
        tangent: &mut NodalStiffnesses<L>,
    ) -> Result<(), String> {
        self.add_stiffness_to(nodal_fields, tangent)
    }
}

impl Load<NodalStatesSolidThermal, NodalForcesSolidThermal> for Convection<Power> {
    fn subtract_from(
        &self,
        nodal_states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        self.check(nodal_states.len())?;
        residual
            .iter_mut()
            .zip(nodal_states.iter().zip(&self.coefficients))
            .for_each(|(residual, (state, &coefficient))| {
                residual.1 += coefficient * (state.1 - self.ambient)
            });
        Ok(())
    }
}

impl LoadStiffness<NodalStatesSolidThermal, NodalStiffnessesSolidThermal> for Convection<Power> {
    fn add_stiffness_to(
        &self,
        nodal_states: &NodalStatesSolidThermal,
        tangent: &mut NodalStiffnessesSolidThermal,
    ) -> Result<(), String> {
        self.check(nodal_states.len())?;
        self.coefficients
            .iter()
            .enumerate()
            .for_each(|(node, &coefficient)| {
                tangent[node][node] += TensorTuple(
                    TensorTuple(Default::default(), Default::default()),
                    TensorTuple(Default::default(), coefficient),
                )
            });
        Ok(())
    }
}

impl LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal> for Convection<Power> {
    fn subtract_at(
        &self,
        _: Quantity<Time>,
        nodal_states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        self.subtract_from(nodal_states, residual)
    }
}

impl LoadStiffnessAt<NodalStatesSolidThermal, NodalStiffnessesSolidThermal> for Convection<Power> {
    fn add_stiffness_at(
        &self,
        _: Quantity<Time>,
        nodal_states: &NodalStatesSolidThermal,
        tangent: &mut NodalStiffnessesSolidThermal,
    ) -> Result<(), String> {
        self.add_stiffness_to(nodal_states, tangent)
    }
}
