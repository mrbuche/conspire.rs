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
    /// A rate, being a reciprocal time.
    Rate,
    /// A viscosity, being a stress per unit rate.
    Viscosity,
    /// A temperature.
    Temperature,
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
    Rate * Viscosity = Stress,
    Length * ReciprocalLength = Dimensionless,
    ReciprocalLength * Length = Dimensionless,
);

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
);
