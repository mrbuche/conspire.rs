#[cfg(test)]
mod test;

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

// Only the products the library actually forms. Reaching for one that is
// absent is a compile error, and the fix is to name the result here.
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
);

// A tuple carries the pair of units its halves do, so the pair combines with
// another the same way, one half at a time.

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
    Stress => ReciprocalStress,
    ReciprocalStress => Stress,
    Temperature => ReciprocalTemperature,
    ReciprocalTemperature => Temperature,
    Viscosity => ReciprocalViscosity,
    ReciprocalViscosity => Viscosity,
    Time => Rate,
    Rate => Time,
);

// Names for units that are the same dimension as one already spelled. An alias
// costs nothing: no product to enumerate, no impl to write, and no
// instantiation of its own, since it denotes the very same type.

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
