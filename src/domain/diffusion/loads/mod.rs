use crate::{
    constitutive::diffusion::{
        Diffusive,
        surface::{SurfaceFlux, SurfaceFluxPerArea, SurfacePotential, SurfaceTangent},
    },
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
    units::{Area, Power, PowerDensity, Time, UnitMul, Volume},
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

/// A flux out of the body through a surface that is a model of the field on it, held as the area
/// that each node has of the surface.
#[derive(Clone, Debug)]
pub struct Surface<S>
where
    S: SurfaceFlux,
{
    areas: Vec<Quantity<Area>>,
    model: S,
}

/// A flux out of the body through a surface that is a model of the field on it, where the model is a
/// function of time, held as the area that each node has of the surface.
pub struct TimedSurface<G> {
    areas: Vec<Quantity<Area>>,
    model: G,
}

type Balance<S> = <S as SurfaceFlux>::Balance;
type Energy<S> = Quantity<<Balance<S> as Diffusive>::Energy>;
type Stiffness<S> = Quantity<<Balance<S> as Diffusive>::Stiffness>;

impl<U, T> Loads<Field<U>, T> {
    /// Adds a flux through a surface, given the area that each node has of it, as a model of the
    /// field.
    pub fn with_surface<S>(
        self,
        areas: Vec<Quantity<Area>>,
        model: S,
    ) -> Loads<Field<U>, (T, Surface<S>)>
    where
        S: SurfaceFlux,
        Balance<S>: Diffusive<Field = U>,
    {
        self.with(Surface { areas, model })
    }
    /// Adds a flux through a surface, given the area that each node has of it, as a model of the
    /// field that is a function of time.
    pub fn with_timed_surface<S, G>(
        self,
        areas: Vec<Quantity<Area>>,
        model: G,
    ) -> Loads<Field<U>, (T, TimedSurface<G>)>
    where
        S: SurfaceFlux,
        Balance<S>: Diffusive<Field = U>,
        G: Fn(Quantity<Time>) -> S,
    {
        self.with(TimedSurface { areas, model })
    }
}

impl<T> Loads<SolidThermal<3>, T> {
    /// Adds a heat flux through a surface, given the area that each node has of it, as a model of
    /// the temperature.
    pub fn with_surface<S>(
        self,
        areas: Vec<Quantity<Area>>,
        model: S,
    ) -> Loads<SolidThermal<3>, (T, Surface<S>)>
    where
        S: SurfaceFlux<Balance = Power>,
    {
        self.with(Surface { areas, model })
    }
    /// Adds a heat flux through a surface, given the area that each node has of it, as a model of
    /// the temperature that is a function of time.
    pub fn with_timed_surface<S, G>(
        self,
        areas: Vec<Quantity<Area>>,
        model: G,
    ) -> Loads<SolidThermal<3>, (T, TimedSurface<G>)>
    where
        S: SurfaceFlux<Balance = Power>,
        G: Fn(Quantity<Time>) -> S,
    {
        self.with(TimedSurface { areas, model })
    }
}

fn check_surface(areas: &[Quantity<Area>], nodes: usize) -> Result<(), String> {
    if nodes == areas.len() {
        Ok(())
    } else {
        Err("The surface is on a mesh with another number of nodes.".to_string())
    }
}

fn subtract_surface<S>(
    areas: &[Quantity<Area>],
    model: &S,
    nodal_fields: &NodalFields<Balance<S>>,
    residual: &mut NodalForces<Balance<S>>,
) -> Result<(), String>
where
    S: SurfaceFlux,
    SurfaceFluxPerArea<S>: Mul<Quantity<Area>, Output = Quantity<Balance<S>>>,
{
    check_surface(areas, nodal_fields.len())?;
    residual
        .iter_mut()
        .zip(nodal_fields.iter().zip(areas))
        .for_each(|(residual, (&field, &area))| *residual += model.flux(field) * area);
    Ok(())
}

fn add_surface_stiffness<S>(
    areas: &[Quantity<Area>],
    model: &S,
    nodal_fields: &NodalFields<Balance<S>>,
    tangent: &mut NodalStiffnesses<Balance<S>>,
) -> Result<(), String>
where
    S: SurfaceFlux,
    SurfaceTangent<S>: Mul<Quantity<Area>, Output = Stiffness<S>>,
{
    check_surface(areas, nodal_fields.len())?;
    nodal_fields
        .iter()
        .zip(areas)
        .enumerate()
        .for_each(|(node, (&field, &area))| tangent[node][node] += model.tangent(field) * area);
    Ok(())
}

fn surface_potential<S>(
    areas: &[Quantity<Area>],
    model: &S,
    nodal_fields: &NodalFields<Balance<S>>,
) -> Result<Energy<S>, String>
where
    S: SurfacePotential,
{
    check_surface(areas, nodal_fields.len())?;
    Ok(nodal_fields
        .iter()
        .zip(areas)
        .map(|(&field, &area)| model.potential(field, area))
        .sum::<Energy<S>>())
}

fn subtract_surface_coupled<S>(
    areas: &[Quantity<Area>],
    model: &S,
    nodal_states: &NodalStatesSolidThermal,
    residual: &mut NodalForcesSolidThermal,
) -> Result<(), String>
where
    S: SurfaceFlux<Balance = Power>,
{
    check_surface(areas, nodal_states.len())?;
    residual
        .iter_mut()
        .zip(nodal_states.iter().zip(areas))
        .for_each(|(residual, (state, &area))| residual.1 += model.flux(state.1) * area);
    Ok(())
}

fn add_surface_stiffness_coupled<S>(
    areas: &[Quantity<Area>],
    model: &S,
    nodal_states: &NodalStatesSolidThermal,
    tangent: &mut NodalStiffnessesSolidThermal,
) -> Result<(), String>
where
    S: SurfaceFlux<Balance = Power>,
{
    check_surface(areas, nodal_states.len())?;
    nodal_states
        .iter()
        .zip(areas)
        .enumerate()
        .for_each(|(node, (state, &area))| {
            tangent[node][node] += TensorTuple(
                TensorTuple(Default::default(), Default::default()),
                TensorTuple(Default::default(), model.tangent(state.1) * area),
            )
        });
    Ok(())
}

impl<S> Load<NodalFields<Balance<S>>, NodalForces<Balance<S>>> for Surface<S>
where
    S: SurfaceFlux,
    SurfaceFluxPerArea<S>: Mul<Quantity<Area>, Output = Quantity<Balance<S>>>,
{
    fn subtract_from(
        &self,
        nodal_fields: &NodalFields<Balance<S>>,
        residual: &mut NodalForces<Balance<S>>,
    ) -> Result<(), String> {
        subtract_surface(&self.areas, &self.model, nodal_fields, residual)
    }
}

impl<S> LoadStiffness<NodalFields<Balance<S>>, NodalStiffnesses<Balance<S>>> for Surface<S>
where
    S: SurfaceFlux,
    SurfaceTangent<S>: Mul<Quantity<Area>, Output = Stiffness<S>>,
{
    fn add_stiffness_to(
        &self,
        nodal_fields: &NodalFields<Balance<S>>,
        tangent: &mut NodalStiffnesses<Balance<S>>,
    ) -> Result<(), String> {
        add_surface_stiffness(&self.areas, &self.model, nodal_fields, tangent)
    }
}

impl<S> LoadAt<NodalFields<Balance<S>>, NodalForces<Balance<S>>> for Surface<S>
where
    S: SurfaceFlux,
    SurfaceFluxPerArea<S>: Mul<Quantity<Area>, Output = Quantity<Balance<S>>>,
{
    fn subtract_at(
        &self,
        _: Quantity<Time>,
        nodal_fields: &NodalFields<Balance<S>>,
        residual: &mut NodalForces<Balance<S>>,
    ) -> Result<(), String> {
        self.subtract_from(nodal_fields, residual)
    }
}

impl<S> LoadStiffnessAt<NodalFields<Balance<S>>, NodalStiffnesses<Balance<S>>> for Surface<S>
where
    S: SurfaceFlux,
    SurfaceTangent<S>: Mul<Quantity<Area>, Output = Stiffness<S>>,
{
    fn add_stiffness_at(
        &self,
        _: Quantity<Time>,
        nodal_fields: &NodalFields<Balance<S>>,
        tangent: &mut NodalStiffnesses<Balance<S>>,
    ) -> Result<(), String> {
        self.add_stiffness_to(nodal_fields, tangent)
    }
}

impl<S> Conservative<NodalFields<Balance<S>>, Energy<S>> for Surface<S>
where
    S: SurfacePotential,
{
    fn potential_energy(
        &self,
        nodal_fields: &NodalFields<Balance<S>>,
    ) -> Result<Energy<S>, String> {
        surface_potential(&self.areas, &self.model, nodal_fields)
    }
}

impl<S> ConservativeAt<NodalFields<Balance<S>>, Energy<S>> for Surface<S>
where
    S: SurfacePotential,
{
    fn potential_energy_at(
        &self,
        _: Quantity<Time>,
        nodal_fields: &NodalFields<Balance<S>>,
    ) -> Result<Energy<S>, String> {
        self.potential_energy(nodal_fields)
    }
}

impl<S> Load<NodalStatesSolidThermal, NodalForcesSolidThermal> for Surface<S>
where
    S: SurfaceFlux<Balance = Power>,
{
    fn subtract_from(
        &self,
        nodal_states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        subtract_surface_coupled(&self.areas, &self.model, nodal_states, residual)
    }
}

impl<S> LoadStiffness<NodalStatesSolidThermal, NodalStiffnessesSolidThermal> for Surface<S>
where
    S: SurfaceFlux<Balance = Power>,
{
    fn add_stiffness_to(
        &self,
        nodal_states: &NodalStatesSolidThermal,
        tangent: &mut NodalStiffnessesSolidThermal,
    ) -> Result<(), String> {
        add_surface_stiffness_coupled(&self.areas, &self.model, nodal_states, tangent)
    }
}

impl<S> LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal> for Surface<S>
where
    S: SurfaceFlux<Balance = Power>,
{
    fn subtract_at(
        &self,
        _: Quantity<Time>,
        nodal_states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        self.subtract_from(nodal_states, residual)
    }
}

impl<S> LoadStiffnessAt<NodalStatesSolidThermal, NodalStiffnessesSolidThermal> for Surface<S>
where
    S: SurfaceFlux<Balance = Power>,
{
    fn add_stiffness_at(
        &self,
        _: Quantity<Time>,
        nodal_states: &NodalStatesSolidThermal,
        tangent: &mut NodalStiffnessesSolidThermal,
    ) -> Result<(), String> {
        self.add_stiffness_to(nodal_states, tangent)
    }
}

impl<S, G> LoadAt<NodalFields<Balance<S>>, NodalForces<Balance<S>>> for TimedSurface<G>
where
    S: SurfaceFlux,
    SurfaceFluxPerArea<S>: Mul<Quantity<Area>, Output = Quantity<Balance<S>>>,
    G: Fn(Quantity<Time>) -> S,
{
    fn subtract_at(
        &self,
        time: Quantity<Time>,
        nodal_fields: &NodalFields<Balance<S>>,
        residual: &mut NodalForces<Balance<S>>,
    ) -> Result<(), String> {
        subtract_surface(&self.areas, &(self.model)(time), nodal_fields, residual)
    }
}

impl<S, G> LoadStiffnessAt<NodalFields<Balance<S>>, NodalStiffnesses<Balance<S>>>
    for TimedSurface<G>
where
    S: SurfaceFlux,
    SurfaceTangent<S>: Mul<Quantity<Area>, Output = Stiffness<S>>,
    G: Fn(Quantity<Time>) -> S,
{
    fn add_stiffness_at(
        &self,
        time: Quantity<Time>,
        nodal_fields: &NodalFields<Balance<S>>,
        tangent: &mut NodalStiffnesses<Balance<S>>,
    ) -> Result<(), String> {
        add_surface_stiffness(&self.areas, &(self.model)(time), nodal_fields, tangent)
    }
}

impl<S, G> ConservativeAt<NodalFields<Balance<S>>, Energy<S>> for TimedSurface<G>
where
    S: SurfacePotential,
    G: Fn(Quantity<Time>) -> S,
{
    fn potential_energy_at(
        &self,
        time: Quantity<Time>,
        nodal_fields: &NodalFields<Balance<S>>,
    ) -> Result<Energy<S>, String> {
        surface_potential(&self.areas, &(self.model)(time), nodal_fields)
    }
}

impl<S, G> LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal> for TimedSurface<G>
where
    S: SurfaceFlux<Balance = Power>,
    G: Fn(Quantity<Time>) -> S,
{
    fn subtract_at(
        &self,
        time: Quantity<Time>,
        nodal_states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        subtract_surface_coupled(&self.areas, &(self.model)(time), nodal_states, residual)
    }
}

impl<S, G> LoadStiffnessAt<NodalStatesSolidThermal, NodalStiffnessesSolidThermal>
    for TimedSurface<G>
where
    S: SurfaceFlux<Balance = Power>,
    G: Fn(Quantity<Time>) -> S,
{
    fn add_stiffness_at(
        &self,
        time: Quantity<Time>,
        nodal_states: &NodalStatesSolidThermal,
        tangent: &mut NodalStiffnessesSolidThermal,
    ) -> Result<(), String> {
        add_surface_stiffness_coupled(&self.areas, &(self.model)(time), nodal_states, tangent)
    }
}
