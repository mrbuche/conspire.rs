//! The scales a quantity may be named in.
//!
//! A unit says what kind of thing a number is; a scale says how big the number
//! is against that kind. Only the kind is worth carrying, so a scale is spent
//! where the quantity is built and the value is held in the base scale
//! thereafter. Every constructor and reader below is a `const fn` over a factor
//! known at compile time, so naming a scale costs nothing at run time.
//!
//! A quantity is read back the way it was written, by naming a scale rather
//! than supplying one, so the base scale is only ever a matter between the
//! quantity and itself.
//!
//! The base scales are SI, coherent throughout: metres, kilograms, seconds and
//! kelvin, so pascals, newtons, joules and watts follow. A factor is only
//! correct against the rest of that set, which is what the tests check by
//! crossing from one unit to another rather than within one.

#[cfg(test)]
mod test;

use super::{
    Area, Dimensionless, Energy, Entropy, Force, ForcePerLength, Length, PowerPerLengthTemperature,
    Rate, ReciprocalTemperature, Stress, StressPerLength, Temperature, Time, Viscosity, Volume,
};
use crate::math::{Quantity, TensorRank0};

/// The kelvin a celsius is measured from.
const ZERO_CELSIUS: TensorRank0 = 273.15;

//
// A reader is named alongside the constructor it undoes rather than derived
// from it, since deriving one identifier from another is what a dependency
// would be for and this crate has none.
//
macro_rules! scales {
    ($($unit:ident { $($name:ident / $reader:ident = $factor:expr, $doc:expr),+ $(,)? })+) => {
        $(
            impl $unit {
                $(
                    #[doc = concat!("A quantity of ", $doc, ".")]
                    pub const fn $name(value: TensorRank0) -> Quantity<$unit> {
                        Quantity::new(value * $factor)
                    }
                )+
            }
            impl Quantity<$unit> {
                $(
                    #[doc = concat!("How many ", $doc, " the quantity is.")]
                    pub const fn $reader(&self) -> TensorRank0 {
                        self.value() / $factor
                    }
                )+
            }
        )+
    };
}

scales!(
    Length {
        meters / in_meters = 1.0, "metres",
        millimeters / in_millimeters = 1e-3, "millimetres",
        micrometers / in_micrometers = 1e-6, "micrometres",
        nanometers / in_nanometers = 1e-9, "nanometres",
        centimeters / in_centimeters = 1e-2, "centimetres",
        kilometers / in_kilometers = 1e3, "kilometres",
        inches / in_inches = 2.54e-2, "inches",
        feet / in_feet = 3.048e-1, "feet",
    }
    Area {
        square_meters / in_square_meters = 1.0, "square metres",
        square_millimeters / in_square_millimeters = 1e-6, "square millimetres",
        square_centimeters / in_square_centimeters = 1e-4, "square centimetres",
    }
    Volume {
        cubic_meters / in_cubic_meters = 1.0, "cubic metres",
        cubic_millimeters / in_cubic_millimeters = 1e-9, "cubic millimetres",
        cubic_centimeters / in_cubic_centimeters = 1e-6, "cubic centimetres",
        liters / in_liters = 1e-3, "litres",
    }
    Time {
        seconds / in_seconds = 1.0, "seconds",
        milliseconds / in_milliseconds = 1e-3, "milliseconds",
        microseconds / in_microseconds = 1e-6, "microseconds",
        minutes / in_minutes = 6e1, "minutes",
        hours / in_hours = 3.6e3, "hours",
    }
    Rate {
        per_second / in_per_second = 1.0, "reciprocal seconds",
        per_minute / in_per_minute = 1.0 / 6e1, "reciprocal minutes",
        per_hour / in_per_hour = 1.0 / 3.6e3, "reciprocal hours",
        hertz / in_hertz = 1.0, "hertz",
    }
    Stress {
        pascals / in_pascals = 1.0, "pascals",
        kilopascals / in_kilopascals = 1e3, "kilopascals",
        megapascals / in_megapascals = 1e6, "megapascals",
        gigapascals / in_gigapascals = 1e9, "gigapascals",
        bars / in_bars = 1e5, "bars",
        psi / in_psi = 6.894_757_293_168_361e3, "pounds per square inch",
        ksi / in_ksi = 6.894_757_293_168_361e6, "kips per square inch",
    }
    Force {
        newtons / in_newtons = 1.0, "newtons",
        millinewtons / in_millinewtons = 1e-3, "millinewtons",
        kilonewtons / in_kilonewtons = 1e3, "kilonewtons",
        meganewtons / in_meganewtons = 1e6, "meganewtons",
        pounds_force / in_pounds_force = 4.448_221_615_260_5, "pounds force",
    }
    Energy {
        joules / in_joules = 1.0, "joules",
        millijoules / in_millijoules = 1e-3, "millijoules",
        kilojoules / in_kilojoules = 1e3, "kilojoules",
        electronvolts / in_electronvolts = 1.602_176_634e-19, "electronvolts",
    }
    Entropy {
        joules_per_kelvin / in_joules_per_kelvin = 1.0, "joules per kelvin",
    }
    ForcePerLength {
        newtons_per_meter / in_newtons_per_meter = 1.0, "newtons per metre",
        piconewtons_per_nanometer / in_piconewtons_per_nanometer = 1e-3,
            "piconewtons per nanometre",
    }
    Viscosity {
        pascal_seconds / in_pascal_seconds = 1.0, "pascal seconds",
        poise / in_poise = 1e-1, "poise",
        centipoise / in_centipoise = 1e-3, "centipoise",
    }
    Temperature {
        kelvin / in_kelvin = 1.0, "kelvin",
    }
    ReciprocalTemperature {
        per_kelvin / in_per_kelvin = 1.0, "reciprocal kelvin",
        per_celsius / in_per_celsius = 1.0, "reciprocal degrees celsius",
    }
    StressPerLength {
        pascals_per_meter / in_pascals_per_meter = 1.0, "pascals per metre",
        megapascals_per_millimeter / in_megapascals_per_millimeter = 1e9,
            "megapascals per millimetre",
    }
    PowerPerLengthTemperature {
        watts_per_meter_kelvin / in_watts_per_meter_kelvin = 1.0, "watts per metre kelvin",
    }
    Dimensionless {
        of / in_ones = 1.0, "no unit at all",
        percent / in_percent = 1e-2, "percent",
    }
);

//
// A celsius is offset from a kelvin rather than scaled, which no factor can
// say and no division can undo. Both directions are written out.
//
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

impl Quantity<Temperature> {
    /// How many degrees celsius the temperature is.
    pub const fn in_celsius(&self) -> TensorRank0 {
        self.value() - ZERO_CELSIUS
    }
    /// How many degrees fahrenheit the temperature is.
    pub const fn in_fahrenheit(&self) -> TensorRank0 {
        (self.value() - ZERO_CELSIUS) * 9.0 / 5.0 + 32.0
    }
}
