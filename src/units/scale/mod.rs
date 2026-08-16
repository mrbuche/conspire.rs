//! The scales a quantity may be named in.
//!
//! A unit says what kind of thing a number is; a scale says how big the number
//! is against that kind. Only the kind is worth carrying, so a scale is spent
//! where the quantity is built and the value is held in the base scale
//! thereafter. Every constructor below is a `const fn` over a factor known at
//! compile time, so naming a scale costs nothing at run time.
//!
//! The base scales are SI, coherent throughout: metres, kilograms, seconds and
//! kelvin, so pascals, newtons, joules and watts follow. A factor is only
//! correct against that set, which is why `physics::molecular` is left out —
//! it counts energy per mole, and a joule and its Boltzmann constant cannot
//! both be right in one `Quantity<Energy>`.

#[cfg(test)]
mod test;

use super::{
    Area, Dimensionless, Force, Length, PowerPerLengthTemperature, Rate, ReciprocalTemperature,
    Stress, StressPerLength, Temperature, Time, Viscosity, Volume,
};
use crate::math::{Quantity, TensorRank0};

/// The kelvin a celsius is measured from.
const ZERO_CELSIUS: TensorRank0 = 273.15;

macro_rules! scales {
    ($($unit:ident { $($name:ident = $factor:expr, $doc:expr),+ $(,)? })+) => {
        $(
            impl $unit {
                $(
                    #[doc = concat!("A quantity of ", $doc, ".")]
                    pub const fn $name(value: TensorRank0) -> Quantity<$unit> {
                        Quantity::new(value * $factor)
                    }
                )+
            }
        )+
    };
}

scales!(
    Length {
        meters = 1.0, "metres",
        millimeters = 1e-3, "millimetres",
        micrometers = 1e-6, "micrometres",
        nanometers = 1e-9, "nanometres",
        centimeters = 1e-2, "centimetres",
        kilometers = 1e3, "kilometres",
        inches = 2.54e-2, "inches",
        feet = 3.048e-1, "feet",
    }
    Area {
        square_meters = 1.0, "square metres",
        square_millimeters = 1e-6, "square millimetres",
        square_centimeters = 1e-4, "square centimetres",
    }
    Volume {
        cubic_meters = 1.0, "cubic metres",
        cubic_millimeters = 1e-9, "cubic millimetres",
        cubic_centimeters = 1e-6, "cubic centimetres",
        liters = 1e-3, "litres",
    }
    Time {
        seconds = 1.0, "seconds",
        milliseconds = 1e-3, "milliseconds",
        microseconds = 1e-6, "microseconds",
        minutes = 6e1, "minutes",
        hours = 3.6e3, "hours",
    }
    Rate {
        per_second = 1.0, "reciprocal seconds",
        per_minute = 1.0 / 6e1, "reciprocal minutes",
        per_hour = 1.0 / 3.6e3, "reciprocal hours",
        hertz = 1.0, "hertz",
    }
    Stress {
        pascals = 1.0, "pascals",
        kilopascals = 1e3, "kilopascals",
        megapascals = 1e6, "megapascals",
        gigapascals = 1e9, "gigapascals",
        bars = 1e5, "bars",
        psi = 6.894_757_293_168_361e3, "pounds per square inch",
        ksi = 6.894_757_293_168_361e6, "kips per square inch",
    }
    Force {
        newtons = 1.0, "newtons",
        millinewtons = 1e-3, "millinewtons",
        kilonewtons = 1e3, "kilonewtons",
        meganewtons = 1e6, "meganewtons",
        pounds_force = 4.448_221_615_260_5, "pounds force",
    }
    Viscosity {
        pascal_seconds = 1.0, "pascal seconds",
        poise = 1e-1, "poise",
        centipoise = 1e-3, "centipoise",
    }
    Temperature {
        kelvin = 1.0, "kelvin",
    }
    ReciprocalTemperature {
        per_kelvin = 1.0, "reciprocal kelvin",
        per_celsius = 1.0, "reciprocal degrees celsius",
    }
    StressPerLength {
        pascals_per_meter = 1.0, "pascals per metre",
        megapascals_per_millimeter = 1e9, "megapascals per millimetre",
    }
    PowerPerLengthTemperature {
        watts_per_meter_kelvin = 1.0, "watts per metre kelvin",
    }
    Dimensionless {
        of = 1.0, "no unit at all",
        percent = 1e-2, "percent",
    }
);

impl Temperature {
    /// A quantity of degrees celsius.
    ///
    /// A celsius is a kelvin offset rather than scaled, so this names a
    /// temperature and not a difference between two. A difference is the same
    /// number of either, and is the [`kelvin`](Self::kelvin) it already is.
    pub const fn celsius(value: TensorRank0) -> Quantity<Temperature> {
        Quantity::new(value + ZERO_CELSIUS)
    }
    /// A quantity of degrees fahrenheit.
    ///
    /// Offset as well as scaled, so the same caution applies as for
    /// [`celsius`](Self::celsius).
    pub const fn fahrenheit(value: TensorRank0) -> Quantity<Temperature> {
        Quantity::new((value - 32.0) * 5.0 / 9.0 + ZERO_CELSIUS)
    }
}
