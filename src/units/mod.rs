//! Dimensional analysis.

#[cfg(test)]
mod test;

mod constants;
mod scale;

pub use constants::{
    AVOGADRO_CONSTANT, BOLTZMANN_CONSTANT, ELEMENTARY_CHARGE, GAS_CONSTANT, LIGHT_SPEED,
    PLANCK_CONSTANT, ROOM_TEMPERATURE, STANDARD_GRAVITY, VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY,
};
pub use scale::{Scale, length_scale};

/// The physical unit a tensor carries.
///
/// Tensors may only be added when their units agree, and multiplying them
/// combines their units through [`UnitMul`], all of which is checked when the
/// operations are compiled rather than when they run.
pub trait Unit {}

/// The unit obtained by multiplying by `Rhs`.
pub trait UnitMul<Rhs> {
    /// The unit of the product.
    type Output;
}

/// The unit obtained by dividing by `Rhs`.
pub trait UnitDiv<Rhs> {
    /// The unit of the quotient.
    type Output;
}

/// A unit that carries no unit, alone or in each part of a pair.
pub trait IsDimensionless {}

impl IsDimensionless for Dimensionless {}

impl<A, B> IsDimensionless for (A, B)
where
    A: IsDimensionless,
    B: IsDimensionless,
{
}

/// A unit whose ratio with itself carries no unit.
pub trait UnitRatio
where
    Self: Sized + UnitDiv<Self>,
{
}

impl<U> UnitRatio for U
where
    U: UnitDiv<U>,
    <U as UnitDiv<U>>::Output: IsDimensionless,
{
}

/// The units the halves of a tuple take from a unit meant for the pair.
///
/// A tuple is scaled either by a step in one variable, which both halves take,
/// or by a quantity whose unit is the pair the halves already carry. Naming both
/// through one trait keeps a single scaling rule for tuples rather than two that
/// would overlap.
pub trait UnitHalves {
    /// The unit the first half takes.
    type First;
    /// The unit the second half takes.
    type Second;
}

impl<A, B> UnitHalves for (A, B) {
    type First = A;
    type Second = B;
}

/// The unit a sum of parts carries, which is the unit each part carries.
///
/// A merit adds the halves of a tuple's contraction together, so a pair whose
/// halves agree is that unit once. Left unimplemented for a pair whose halves
/// differ, since adding those names nothing. Implemented concretely rather than
/// blanketly for the same reason [`UnitHalves`] is: a blanket identity would
/// overlap the pair.
pub trait UnitSum {
    /// The unit of the sum.
    type Output;
}

impl<A> UnitSum for (A, A) {
    type Output = A;
}

/// The unit obtained by inverting.
///
/// Left unimplemented for units whose inverse names no quantity, so that
/// inverting such a tensor fails to compile.
pub trait UnitInv {
    /// The unit of the inverse.
    type Output;
}

/// The unit whose square this unit is, which a square root carries.
///
/// Left unimplemented for units that are no square this library names, so that
/// taking the root of such a quantity fails to compile.
pub trait UnitRoot {
    /// The unit of the square root.
    type Output;
}

macro_rules! units {
    ($($(#[$meta:meta])* $name:ident),+ $(,)?) => {
        $(
            $(#[$meta])*
            #[derive(Clone, Copy, Debug, PartialEq, Eq)]
            pub struct $name;
            impl Unit for $name {}
            impl UnitMul<Dimensionless> for $name {
                type Output = $name;
            }
            impl UnitDiv<Dimensionless> for $name {
                type Output = $name;
            }
            impl UnitHalves for $name {
                type First = $name;
                type Second = $name;
            }
            impl UnitSum for $name {
                type Output = $name;
            }
        )+
    };
}

units!(
    /// A quantity carrying no unit.
    Dimensionless,
    /// A length.
    Length,
    /// A reciprocal length.
    ReciprocalLength,
    /// An area.
    Area,
    /// A reciprocal area.
    ReciprocalArea,
    /// A volume.
    Volume,
    /// A second moment of area, being an area squared.
    SecondMomentOfArea,
    /// A velocity, being a length per unit time.
    Velocity,
    /// An acceleration, being a velocity per unit time.
    Acceleration,
    /// A mass.
    Mass,
    /// A density, being a mass per unit volume.
    Density,
    /// A specific energy, being an energy per unit mass, and equally a squared velocity.
    SpecificEnergy,
    /// A specific entropy, being an entropy per unit mass, as a specific heat is.
    SpecificEntropy,
    /// A force.
    Force,
    /// A force per unit length, as a stiffness is.
    ForcePerLength,
    /// A force per unit velocity, as a damping is.
    ForcePerVelocity,
    /// An energy.
    Energy,
    /// A power.
    Power,
    /// A stress per unit length, as a force per unit volume is.
    StressPerLength,
    /// A stress per unit area, as a stiffness per unit volume is.
    StressPerArea,
    /// A viscosity per unit length.
    ViscosityPerLength,
    /// A viscosity per unit area, as a damping per unit volume is.
    ViscosityPerArea,
    /// A stress, and equally a stiffness, being a stress per unit strain.
    Stress,
    /// A squared stress.
    StressSquared,
    /// A reciprocal stress, as a compliance is.
    ReciprocalStress,
    /// A time.
    Time,
    /// A rate, being a reciprocal time.
    Rate,
    /// A squared rate, being a squared reciprocal time.
    RateSquared,
    /// A viscosity, being a stress per unit rate.
    Viscosity,
    /// A fluidity, being the reciprocal of a viscosity.
    ReciprocalViscosity,
    /// A temperature.
    Temperature,
    /// A reciprocal temperature, as a coefficient of thermal expansion is.
    ReciprocalTemperature,
    /// A reciprocal temperature squared, as the change of a coefficient of thermal expansion with temperature is.
    ReciprocalTemperatureSquared,
    /// A time per unit temperature.
    TimePerTemperature,
    /// A temperature per unit time, as a rate of heating is.
    TemperatureRate,
    /// A stress per unit temperature, as a thermal stress coefficient is.
    StressPerTemperature,
    /// A stress per unit temperature squared, as the change of a thermal stress coefficient with temperature is.
    StressPerTemperatureSquared,
    /// A stress per unit length per unit temperature, as a thermal stress coefficient per unit length is.
    StressPerLengthTemperature,
    /// A force per unit temperature, as the dependence of a nodal force on a nodal temperature is.
    ForcePerTemperature,
    /// A power per unit volume, as a dissipation is.
    PowerDensity,
    /// A temperature per unit length, as a temperature gradient is.
    TemperaturePerLength,
    /// A squared temperature gradient, being a squared temperature per unit area.
    TemperatureSquaredPerArea,
    /// A power per unit area, as a heat flux is.
    PowerPerArea,
    /// A power per unit length per unit temperature, as a thermal conductivity is.
    PowerPerLengthTemperature,
    /// A power per unit area per unit temperature, as a heat flux per unit temperature is.
    PowerPerAreaTemperature,
    /// A power per unit volume per unit temperature, as an entropy production rate per unit volume is.
    PowerPerVolumeTemperature,
    /// A power per unit volume per unit temperature squared, as the change of an entropy production rate per unit volume with temperature is.
    PowerPerVolumeTemperatureSquared,
    /// A power per unit length, as the dependence of a thermal power on a nodal displacement is.
    PowerPerLength,
    /// A power per unit temperature, as a thermal stiffness is.
    PowerPerTemperature,
    /// A power times a temperature, as the potential a heat flux derives from is.
    PowerTemperature,
    /// A power times a temperature per unit volume.
    PowerTemperatureDensity,
    /// An entropy, and equally the Boltzmann constant, being an energy per unit temperature.
    Entropy,
    /// An action, being an energy times a time, as the Planck constant is.
    Action,
    /// An amount of substance.
    Amount,
    /// A reciprocal amount of substance, as the Avogadro constant is.
    ReciprocalAmount,
    /// A molar entropy, as the gas constant is, being an entropy per unit amount.
    MolarEntropy,
    /// A molar energy, being an energy per unit amount.
    MolarEnergy,
    /// An electric charge, as the elementary charge is.
    Charge,
    /// An electric potential, being an energy per unit charge.
    ElectricPotential,
    /// An electric field, being an electric potential per unit length.
    ElectricField,
    /// A squared electric field, being a squared electric potential per unit area.
    ElectricFieldSquared,
    /// An electric displacement, being a charge per unit area.
    ElectricDisplacement,
    /// A permittivity, being a capacitance per unit length.
    Permittivity,
    /// A permeability, being an inductance per unit length.
    Permeability,
    /// A capacitance, being a charge per unit electric potential.
    Capacitance,
    /// A capacitance per unit area, as a permittivity per unit length is.
    CapacitancePerArea,
    /// A capacitance per unit volume, as a permittivity per unit area is.
    CapacitancePerVolume,
    /// A charge per unit volume, as a free charge density is.
    ChargeDensity,
    /// An electric current, being a charge per unit time.
    Current,
    /// A current density, being a current per unit area.
    CurrentDensity,
    /// A current per unit volume, as a current density per unit length is.
    CurrentPerVolume,
    /// A conductance, being a current per unit electric potential.
    Conductance,
    /// A conductivity, being a conductance per unit length.
    Conductivity,
    /// A conductance per unit area, as a conductivity per unit length is.
    ConductancePerArea,
    /// A conductance per unit volume, as a conductivity per unit area is.
    ConductancePerVolume,
    /// A reciprocal stiffness, as a compliance conjugate to a force is.
    ReciprocalForcePerLength,
);

macro_rules! unit_products {
    ($($lhs:ident * $rhs:ident = $out:ident),+ $(,)?) => {
        $(
            impl UnitMul<$rhs> for $lhs {
                type Output = $out;
            }
            impl UnitDiv<$rhs> for $out {
                type Output = $lhs;
            }
        )+
    };
}

impl UnitRoot for Dimensionless {
    type Output = Dimensionless;
}

impl UnitRoot for RateSquared {
    type Output = Rate;
}

impl UnitRoot for StressSquared {
    type Output = Stress;
}

impl UnitRoot for SpecificEnergy {
    type Output = Velocity;
}

unit_products!(
    Dimensionless * Length = Length,
    Dimensionless * ReciprocalLength = ReciprocalLength,
    Dimensionless * Stress = Stress,
    Dimensionless * Rate = Rate,
    Dimensionless * Viscosity = Viscosity,
    Dimensionless * Temperature = Temperature,
    Viscosity * Rate = Stress,
    Dimensionless * ReciprocalViscosity = ReciprocalViscosity,
    Stress * ReciprocalViscosity = Rate,
    ReciprocalViscosity * Stress = Rate,
    Rate * Viscosity = Stress,
    Length * ReciprocalLength = Dimensionless,
    Dimensionless * ReciprocalStress = ReciprocalStress,
    Stress * ReciprocalStress = Dimensionless,
    ReciprocalStress * Stress = Dimensionless,
    Dimensionless * ReciprocalTemperature = ReciprocalTemperature,
    Dimensionless * ReciprocalTemperatureSquared = ReciprocalTemperatureSquared,
    ReciprocalTemperature * ReciprocalTemperature = ReciprocalTemperatureSquared,
    ReciprocalTemperatureSquared * Temperature = ReciprocalTemperature,
    Stress * ReciprocalTemperatureSquared = StressPerTemperatureSquared,
    Dimensionless * TimePerTemperature = TimePerTemperature,
    Time * ReciprocalTemperature = TimePerTemperature,
    ReciprocalTemperature * Time = TimePerTemperature,
    TimePerTemperature * Temperature = Time,
    Temperature * TimePerTemperature = Time,
    TimePerTemperature * Power = Entropy,
    Power * TimePerTemperature = Entropy,
    TimePerTemperature * PowerTemperature = Energy,
    PowerTemperature * TimePerTemperature = Energy,
    TimePerTemperature * PowerTemperatureDensity = Stress,
    PowerTemperatureDensity * TimePerTemperature = Stress,
    Temperature * ReciprocalTemperature = Dimensionless,
    Dimensionless * StressPerTemperature = StressPerTemperature,
    Stress * ReciprocalTemperature = StressPerTemperature,
    StressPerTemperature * Temperature = Stress,
    Dimensionless * StressPerTemperatureSquared = StressPerTemperatureSquared,
    StressPerTemperature * ReciprocalTemperature = StressPerTemperatureSquared,
    StressPerTemperatureSquared * Temperature = StressPerTemperature,
    Dimensionless * StressPerLengthTemperature = StressPerLengthTemperature,
    StressPerTemperature * ReciprocalLength = StressPerLengthTemperature,
    ReciprocalLength * StressPerTemperature = StressPerLengthTemperature,
    StressPerLengthTemperature * Volume = ForcePerTemperature,
    Dimensionless * ForcePerTemperature = ForcePerTemperature,
    ForcePerTemperature * Temperature = Force,
    ReciprocalTemperature * Temperature = Dimensionless,
    ReciprocalLength * Length = Dimensionless,
    Dimensionless * Time = Time,
    Rate * Time = Dimensionless,
    Time * Rate = Dimensionless,
    Dimensionless * RateSquared = RateSquared,
    Rate * Rate = RateSquared,
    Dimensionless * StressSquared = StressSquared,
    Stress * Stress = StressSquared,
    Stress * Time = Viscosity,
    Dimensionless * PowerDensity = PowerDensity,
    Stress * Rate = PowerDensity,
    Rate * Stress = PowerDensity,
    Dimensionless * Area = Area,
    Dimensionless * ReciprocalArea = ReciprocalArea,
    Dimensionless * Volume = Volume,
    Dimensionless * SecondMomentOfArea = SecondMomentOfArea,
    Dimensionless * Velocity = Velocity,
    Dimensionless * Force = Force,
    Dimensionless * ForcePerLength = ForcePerLength,
    Dimensionless * ForcePerVelocity = ForcePerVelocity,
    Dimensionless * Energy = Energy,
    Dimensionless * Power = Power,
    Dimensionless * StressPerLength = StressPerLength,
    Dimensionless * StressPerArea = StressPerArea,
    Dimensionless * ViscosityPerLength = ViscosityPerLength,
    Dimensionless * ViscosityPerArea = ViscosityPerArea,
    Length * Length = Area,
    Length * Area = Volume,
    Area * Length = Volume,
    Area * Area = SecondMomentOfArea,
    Length * Volume = SecondMomentOfArea,
    Volume * Length = SecondMomentOfArea,
    Area * ReciprocalArea = Dimensionless,
    ReciprocalArea * Area = Dimensionless,
    Length * ReciprocalArea = ReciprocalLength,
    ReciprocalArea * Length = ReciprocalLength,
    ReciprocalLength * ReciprocalLength = ReciprocalArea,
    ReciprocalLength * Volume = Area,
    ReciprocalLength * Area = Length,
    Area * ReciprocalLength = Length,
    Length * Rate = Velocity,
    Rate * Length = Velocity,
    Velocity * Time = Length,
    Velocity * ReciprocalLength = Rate,
    ReciprocalLength * Velocity = Rate,
    Stress * Area = Force,
    Area * Stress = Force,
    Stress * Volume = Energy,
    Force * Length = Energy,
    Force * ReciprocalLength = ForcePerLength,
    ForcePerLength * Length = Force,
    Length * ForcePerLength = Force,
    Velocity * ForcePerVelocity = Force,
    Stress * ReciprocalLength = StressPerLength,
    ReciprocalLength * Stress = StressPerLength,
    StressPerLength * ReciprocalLength = StressPerArea,
    StressPerLength * Length = Stress,
    Length * StressPerLength = Stress,
    StressPerLength * Area = ForcePerLength,
    StressPerLength * Volume = Force,
    StressPerArea * Volume = ForcePerLength,
    Viscosity * ReciprocalLength = ViscosityPerLength,
    ReciprocalLength * Viscosity = ViscosityPerLength,
    ViscosityPerLength * ReciprocalLength = ViscosityPerArea,
    ViscosityPerArea * Volume = ForcePerVelocity,
    ForcePerVelocity * Velocity = Force,
    Force * Velocity = Power,
    PowerDensity * Volume = Power,
    Energy * Rate = Power,
    Power * Time = Energy,
    Dimensionless * PowerPerLength = PowerPerLength,
    Power * ReciprocalLength = PowerPerLength,
    Force * Rate = PowerPerLength,
    Rate * Force = PowerPerLength,
    ForcePerLength * Velocity = PowerPerLength,
    Velocity * ForcePerLength = PowerPerLength,
    PowerPerLength * Length = Power,
    Dimensionless * TemperaturePerLength = TemperaturePerLength,
    Dimensionless * TemperatureSquaredPerArea = TemperatureSquaredPerArea,
    TemperaturePerLength * TemperaturePerLength = TemperatureSquaredPerArea,
    TemperatureSquaredPerArea * PowerPerLengthTemperature = PowerTemperatureDensity,
    PowerPerLengthTemperature * TemperatureSquaredPerArea = PowerTemperatureDensity,
    Dimensionless * PowerPerArea = PowerPerArea,
    Dimensionless * PowerPerLengthTemperature = PowerPerLengthTemperature,
    Dimensionless * PowerPerAreaTemperature = PowerPerAreaTemperature,
    Dimensionless * PowerPerVolumeTemperature = PowerPerVolumeTemperature,
    Dimensionless * PowerPerTemperature = PowerPerTemperature,
    Dimensionless * PowerTemperature = PowerTemperature,
    Dimensionless * PowerTemperatureDensity = PowerTemperatureDensity,
    Temperature * ReciprocalLength = TemperaturePerLength,
    ReciprocalLength * Temperature = TemperaturePerLength,
    TemperaturePerLength * PowerPerLengthTemperature = PowerPerArea,
    PowerPerLengthTemperature * TemperaturePerLength = PowerPerArea,
    PowerPerArea * ReciprocalLength = PowerDensity,
    ReciprocalLength * PowerPerArea = PowerDensity,
    PowerPerArea * TemperaturePerLength = PowerTemperatureDensity,
    PowerTemperatureDensity * Volume = PowerTemperature,
    PowerPerLengthTemperature * ReciprocalLength = PowerPerAreaTemperature,
    ReciprocalLength * PowerPerLengthTemperature = PowerPerAreaTemperature,
    PowerPerAreaTemperature * ReciprocalLength = PowerPerVolumeTemperature,
    ReciprocalLength * PowerPerAreaTemperature = PowerPerVolumeTemperature,
    PowerDensity * ReciprocalTemperature = EntropyDensityRate,
    EntropyDensity * Rate = EntropyDensityRate,
    Dimensionless * PowerPerVolumeTemperatureSquared = PowerPerVolumeTemperatureSquared,
    StressPerTemperatureSquared * Rate = PowerPerVolumeTemperatureSquared,
    Rate * StressPerTemperatureSquared = PowerPerVolumeTemperatureSquared,
    PowerPerVolumeTemperatureSquared * Temperature = PowerPerVolumeTemperature,
    PowerPerVolumeTemperature * Volume = PowerPerTemperature,
    Power * Temperature = PowerTemperature,
    PowerPerTemperature * Temperature = Power,
    Temperature * PowerPerTemperature = Power,
    Dimensionless * TemperatureRate = TemperatureRate,
    Temperature * Rate = TemperatureRate,
    Rate * Temperature = TemperatureRate,
    TemperatureRate * Time = Temperature,
    Time * TemperatureRate = Temperature,
    TemperatureRate * Entropy = Power,
    Entropy * TemperatureRate = Power,
    Dimensionless * Entropy = Entropy,
    Entropy * Temperature = Energy,
    Temperature * Entropy = Energy,
    Dimensionless * Action = Action,
    Dimensionless * Amount = Amount,
    Dimensionless * ReciprocalAmount = ReciprocalAmount,
    Dimensionless * MolarEntropy = MolarEntropy,
    Dimensionless * MolarEnergy = MolarEnergy,
    Dimensionless * Charge = Charge,
    Dimensionless * ElectricPotential = ElectricPotential,
    Dimensionless * ElectricField = ElectricField,
    Dimensionless * ElectricFieldSquared = ElectricFieldSquared,
    ElectricField * ElectricField = ElectricFieldSquared,
    Permittivity * ElectricFieldSquared = Stress,
    ElectricFieldSquared * Permittivity = Stress,
    Dimensionless * ElectricDisplacement = ElectricDisplacement,
    Dimensionless * Permittivity = Permittivity,
    Dimensionless * Permeability = Permeability,
    Dimensionless * Capacitance = Capacitance,
    Dimensionless * CapacitancePerArea = CapacitancePerArea,
    Dimensionless * CapacitancePerVolume = CapacitancePerVolume,
    Dimensionless * ChargeDensity = ChargeDensity,
    ElectricPotential * Charge = Energy,
    Charge * ElectricPotential = Energy,
    Capacitance * ElectricPotential = Charge,
    ElectricPotential * Capacitance = Charge,
    ElectricPotential * ReciprocalLength = ElectricField,
    ReciprocalLength * ElectricPotential = ElectricField,
    ElectricField * Length = ElectricPotential,
    Length * ElectricField = ElectricPotential,
    Permittivity * ElectricField = ElectricDisplacement,
    ElectricField * Permittivity = ElectricDisplacement,
    Permittivity * Length = Capacitance,
    Length * Permittivity = Capacitance,
    Permittivity * ReciprocalLength = CapacitancePerArea,
    ReciprocalLength * Permittivity = CapacitancePerArea,
    CapacitancePerArea * ReciprocalLength = CapacitancePerVolume,
    ReciprocalLength * CapacitancePerArea = CapacitancePerVolume,
    CapacitancePerArea * Area = Capacitance,
    Area * CapacitancePerArea = Capacitance,
    CapacitancePerVolume * Volume = Capacitance,
    Volume * CapacitancePerVolume = Capacitance,
    ElectricDisplacement * Area = Charge,
    Area * ElectricDisplacement = Charge,
    ElectricDisplacement * ReciprocalLength = ChargeDensity,
    ReciprocalLength * ElectricDisplacement = ChargeDensity,
    ChargeDensity * Volume = Charge,
    Volume * ChargeDensity = Charge,
    ElectricField * ElectricDisplacement = Stress,
    ElectricDisplacement * ElectricField = Stress,
    Dimensionless * Current = Current,
    Dimensionless * CurrentDensity = CurrentDensity,
    Dimensionless * CurrentPerVolume = CurrentPerVolume,
    Dimensionless * Conductance = Conductance,
    Dimensionless * Conductivity = Conductivity,
    Dimensionless * ConductancePerArea = ConductancePerArea,
    Dimensionless * ConductancePerVolume = ConductancePerVolume,
    Current * Time = Charge,
    Time * Current = Charge,
    Current * ElectricPotential = Power,
    ElectricPotential * Current = Power,
    Conductance * ElectricPotential = Current,
    ElectricPotential * Conductance = Current,
    Conductivity * ElectricField = CurrentDensity,
    ElectricField * Conductivity = CurrentDensity,
    Conductivity * Length = Conductance,
    Length * Conductivity = Conductance,
    Conductivity * ReciprocalLength = ConductancePerArea,
    ReciprocalLength * Conductivity = ConductancePerArea,
    ConductancePerArea * ReciprocalLength = ConductancePerVolume,
    ReciprocalLength * ConductancePerArea = ConductancePerVolume,
    ConductancePerArea * Area = Conductance,
    Area * ConductancePerArea = Conductance,
    ConductancePerVolume * Volume = Conductance,
    Volume * ConductancePerVolume = Conductance,
    CurrentDensity * Area = Current,
    Area * CurrentDensity = Current,
    CurrentDensity * ReciprocalLength = CurrentPerVolume,
    ReciprocalLength * CurrentDensity = CurrentPerVolume,
    CurrentPerVolume * Volume = Current,
    Volume * CurrentPerVolume = Current,
    Conductivity * ElectricFieldSquared = PowerDensity,
    ElectricFieldSquared * Conductivity = PowerDensity,
    ElectricField * CurrentDensity = PowerDensity,
    CurrentDensity * ElectricField = PowerDensity,
    Energy * Time = Action,
    Time * Energy = Action,
    Action * Rate = Energy,
    Rate * Action = Energy,
    Amount * ReciprocalAmount = Dimensionless,
    ReciprocalAmount * Amount = Dimensionless,
    Entropy * ReciprocalAmount = MolarEntropy,
    ReciprocalAmount * Entropy = MolarEntropy,
    MolarEntropy * Amount = Entropy,
    MolarEntropy * Temperature = MolarEnergy,
    Temperature * MolarEntropy = MolarEnergy,
    Energy * ReciprocalAmount = MolarEnergy,
    ReciprocalAmount * Energy = MolarEnergy,
    MolarEnergy * Amount = Energy,
    Dimensionless * ReciprocalForcePerLength = ReciprocalForcePerLength,
    ReciprocalLength * Energy = Force,
    Energy * ReciprocalLength = Force,
    ReciprocalArea * Energy = ForcePerLength,
    Energy * ReciprocalArea = ForcePerLength,
    ForcePerLength * ReciprocalLength = Stress,
    ReciprocalLength * ForcePerLength = Stress,
    ForcePerLength * Area = Energy,
    Area * ForcePerLength = Energy,
    ReciprocalForcePerLength * Energy = Area,
    Energy * ReciprocalForcePerLength = Area,
    Length * Force = Energy,
    ReciprocalForcePerLength * Force = Length,
    Force * ReciprocalForcePerLength = Length,
    Stress * Length = ForcePerLength,
    Length * Stress = ForcePerLength,
    Dimensionless * Acceleration = Acceleration,
    Dimensionless * Mass = Mass,
    Dimensionless * Density = Density,
    Velocity * Rate = Acceleration,
    Rate * Velocity = Acceleration,
    Length * RateSquared = Acceleration,
    RateSquared * Length = Acceleration,
    Acceleration * Time = Velocity,
    Time * Acceleration = Velocity,
    Density * Volume = Mass,
    Volume * Density = Mass,
    Mass * Acceleration = Force,
    Acceleration * Mass = Force,
    Mass * RateSquared = ForcePerLength,
    RateSquared * Mass = ForcePerLength,
    Density * Acceleration = StressPerLength,
    Acceleration * Density = StressPerLength,
    Dimensionless * SpecificEnergy = SpecificEnergy,
    Velocity * Velocity = SpecificEnergy,
    Mass * SpecificEnergy = Energy,
    SpecificEnergy * Mass = Energy,
    Density * SpecificEnergy = Stress,
    SpecificEnergy * Density = Stress,
    Dimensionless * SpecificEntropy = SpecificEntropy,
    Mass * SpecificEntropy = Entropy,
    SpecificEntropy * Mass = Entropy,
    Density * SpecificEntropy = StressPerTemperature,
    SpecificEntropy * Density = StressPerTemperature,
    StressPerTemperature * Volume = Entropy,
    Volume * StressPerTemperature = Entropy,
);

impl<A, B, C, D> UnitMul<(C, D)> for (A, B)
where
    A: UnitMul<C>,
    B: UnitMul<D>,
{
    type Output = (<A as UnitMul<C>>::Output, <B as UnitMul<D>>::Output);
}

impl<A, B, C, D> UnitDiv<(C, D)> for (A, B)
where
    A: UnitDiv<C>,
    B: UnitDiv<D>,
{
    type Output = (<A as UnitDiv<C>>::Output, <B as UnitDiv<D>>::Output);
}

macro_rules! unit_inverses {
    ($($unit:ident => $inverse:ident),+ $(,)?) => {
        $(
            impl UnitInv for $unit {
                type Output = $inverse;
            }
        )+
    };
}

unit_inverses!(
    Dimensionless => Dimensionless,
    Length => ReciprocalLength,
    ReciprocalLength => Length,
    Area => ReciprocalArea,
    ReciprocalArea => Area,
    Stress => ReciprocalStress,
    ReciprocalStress => Stress,
    Temperature => ReciprocalTemperature,
    ReciprocalTemperature => Temperature,
    Viscosity => ReciprocalViscosity,
    ReciprocalViscosity => Viscosity,
    Time => Rate,
    Rate => Time,
    Amount => ReciprocalAmount,
    ReciprocalAmount => Amount,
    ForcePerLength => ReciprocalForcePerLength,
    ReciprocalForcePerLength => ForcePerLength,
);

/// A pressure.
pub type Pressure = Stress;

/// An energy per unit volume.
pub type EnergyDensity = Stress;

/// An elastic modulus.
pub type Modulus = Stress;

/// A compliance.
pub type Compliance = ReciprocalStress;

/// A fluidity.
pub type Fluidity = ReciprocalViscosity;

/// A coefficient of thermal expansion.
pub type ThermalExpansion = ReciprocalTemperature;

/// A frequency.
pub type Frequency = Rate;

/// A dissipation, being a power per unit volume.
pub type Dissipation = PowerDensity;

/// An entropy per unit volume.
pub type EntropyDensity = StressPerTemperature;

/// A specific heat, being a heat capacity per unit mass.
pub type SpecificHeat = SpecificEntropy;

/// A heat capacity.
pub type HeatCapacity = Entropy;

/// A heat capacity per unit volume.
pub type VolumetricHeatCapacity = StressPerTemperature;

/// An entropy production rate per unit volume.
pub type EntropyDensityRate = PowerPerVolumeTemperature;
