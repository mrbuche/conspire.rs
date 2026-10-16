//! Diffusion constitutive models.

use std::fmt::Debug;

use crate::{
    constitutive::ConstitutiveError,
    math::{Quantity, Reference, TensorRank1, TensorRank1List, TensorRank2, TensorRank2List},
    units::{
        Capacitance, CapacitancePerArea, CapacitancePerVolume, Charge, ChargeDensity,
        ElectricDisplacement, ElectricField, ElectricPotential, Energy, EnergyDensity,
        Permittivity, Power, PowerDensity, PowerPerArea, PowerPerAreaTemperature,
        PowerPerLengthTemperature, PowerPerTemperature, PowerPerVolumeTemperature,
        PowerTemperature, PowerTemperatureDensity, ReciprocalLength, Temperature,
        TemperaturePerLength, Unit, UnitMul, Volume,
    },
};

/// A potential whose gradient drives a flux, and the units that follow from it.
pub trait Diffusive
where
    Self: Unit,
{
    /// The unit of the gradient of the potential.
    type Gradient: Unit;
    /// The unit of the flux.
    type Flux: Unit + UnitMul<ReciprocalLength, Output = Self::FluxPerLength>;
    /// The unit of the flux per length.
    type FluxPerLength: Unit + UnitMul<Volume, Output = Self::Force>;
    /// The unit of the tangent of the flux.
    type Tangent: Unit + UnitMul<ReciprocalLength, Output = Self::TangentPerLength>;
    /// The unit of the tangent per length.
    type TangentPerLength: Unit + UnitMul<ReciprocalLength, Output = Self::TangentPerArea>;
    /// The unit of the tangent per area.
    type TangentPerArea: Unit + UnitMul<Volume, Output = Self::Stiffness>;
    /// The unit of the energy density.
    type EnergyDensity: Unit + UnitMul<Volume, Output = Self::Energy>;
    /// The unit of the energy.
    type Energy: Unit;
    /// The unit of the nodal force.
    type Force: Unit;
    /// The unit of the nodal stiffness.
    type Stiffness: Unit;
}

impl Diffusive for Temperature {
    type Gradient = TemperaturePerLength;
    type Flux = PowerPerArea;
    type FluxPerLength = PowerDensity;
    type Tangent = PowerPerLengthTemperature;
    type TangentPerLength = PowerPerAreaTemperature;
    type TangentPerArea = PowerPerVolumeTemperature;
    type EnergyDensity = PowerTemperatureDensity;
    type Energy = PowerTemperature;
    type Force = Power;
    type Stiffness = PowerPerTemperature;
}

impl Diffusive for ElectricPotential {
    type Gradient = ElectricField;
    type Flux = ElectricDisplacement;
    type FluxPerLength = ChargeDensity;
    type Tangent = Permittivity;
    type TangentPerLength = CapacitancePerArea;
    type TangentPerArea = CapacitancePerVolume;
    type EnergyDensity = EnergyDensity;
    type Energy = Energy;
    type Force = Charge;
    type Stiffness = Capacitance;
}

/// The potential of a diffusion constitutive model.
pub type Potential<C> = <C as Diffusion>::Potential;

/// The gradient of the potential.
pub type DiffusionGradient<C> =
    TensorRank1<3, Reference, <<C as Diffusion>::Potential as Diffusive>::Gradient>;

/// A list of gradients of the potential.
pub type DiffusionGradients<C, const N: usize> =
    TensorRank1List<3, Reference, N, <<C as Diffusion>::Potential as Diffusive>::Gradient>;

/// The flux.
pub type DiffusionFlux<C> =
    TensorRank1<3, Reference, <<C as Diffusion>::Potential as Diffusive>::Flux>;

/// A list of fluxes.
pub type DiffusionFluxes<C, const N: usize> =
    TensorRank1List<3, Reference, N, <<C as Diffusion>::Potential as Diffusive>::Flux>;

/// The tangent to the flux.
pub type DiffusionTangent<C> =
    TensorRank2<3, Reference, Reference, <<C as Diffusion>::Potential as Diffusive>::Tangent>;

/// A list of tangents to the flux.
pub type DiffusionTangents<C, const N: usize> = TensorRank2List<
    3,
    Reference,
    Reference,
    N,
    <<C as Diffusion>::Potential as Diffusive>::Tangent,
>;

/// Required methods for diffusion constitutive models.
///
/// The flux is $`-\kappa\nabla\phi`$ for a potential $`\phi`$, so that the
/// tangent is negative and the potential is the energy density whose
/// derivative with respect to the gradient is the negative flux.
pub trait Diffusion
where
    Self: Clone + Debug,
{
    /// The potential whose gradient drives the flux.
    type Potential: Diffusive;
    /// Calculates and returns the potential.
    fn potential(
        &self,
        gradient: &DiffusionGradient<Self>,
    ) -> Result<Quantity<<Self::Potential as Diffusive>::EnergyDensity>, ConstitutiveError>;
    /// Calculates and returns the flux.
    fn flux(
        &self,
        gradient: &DiffusionGradient<Self>,
    ) -> Result<DiffusionFlux<Self>, ConstitutiveError>;
    /// Calculates and returns the tangent to the flux.
    fn tangent(
        &self,
        gradient: &DiffusionGradient<Self>,
    ) -> Result<DiffusionTangent<Self>, ConstitutiveError>;
}
