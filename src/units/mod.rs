//! Dimensional analysis.

#[cfg(test)]
mod test;

#[cfg(feature = "math")]
mod scale;

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
    /// A reciprocal stress, as a compliance is.
    ReciprocalStress,
    /// A time.
    Time,
    /// A rate, being a reciprocal time.
    Rate,
    /// A viscosity, being a stress per unit rate.
    Viscosity,
    /// A fluidity, being the reciprocal of a viscosity.
    ReciprocalViscosity,
    /// A temperature.
    Temperature,
    /// A reciprocal temperature, as a coefficient of thermal expansion is.
    ReciprocalTemperature,
    /// A stress per unit temperature, as a thermal stress coefficient is.
    StressPerTemperature,
    /// A power per unit volume, as a dissipation is.
    PowerDensity,
    /// A temperature per unit length, as a temperature gradient is.
    TemperaturePerLength,
    /// A power per unit area, as a heat flux is.
    PowerPerArea,
    /// A power per unit length per unit temperature, as a thermal conductivity is.
    PowerPerLengthTemperature,
    /// A power per unit area per unit temperature, as a heat flux per unit temperature is.
    PowerPerAreaTemperature,
    /// A power per unit volume per unit temperature.
    PowerPerVolumeTemperature,
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
    Temperature * ReciprocalTemperature = Dimensionless,
    Dimensionless * StressPerTemperature = StressPerTemperature,
    Stress * ReciprocalTemperature = StressPerTemperature,
    StressPerTemperature * Temperature = Stress,
    ReciprocalTemperature * Temperature = Dimensionless,
    ReciprocalLength * Length = Dimensionless,
    Dimensionless * Time = Time,
    Rate * Time = Dimensionless,
    Time * Rate = Dimensionless,
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
    Dimensionless * TemperaturePerLength = TemperaturePerLength,
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
    PowerPerVolumeTemperature * Volume = PowerPerTemperature,
    Power * Temperature = PowerTemperature,
    PowerPerTemperature * Temperature = Power,
    Temperature * PowerPerTemperature = Power,
    Dimensionless * Entropy = Entropy,
    Entropy * Temperature = Energy,
    Temperature * Entropy = Energy,
    Dimensionless * Action = Action,
    Dimensionless * Amount = Amount,
    Dimensionless * ReciprocalAmount = ReciprocalAmount,
    Dimensionless * MolarEntropy = MolarEntropy,
    Dimensionless * MolarEnergy = MolarEnergy,
    Dimensionless * Charge = Charge,
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
