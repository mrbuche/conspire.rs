//! The defining constants.

#[cfg(test)]
mod test;

use super::{
    Acceleration, Action, Charge, Entropy, MolarEntropy, Permeability, Permittivity,
    PowerPerAreaTemperatureFourth, ReciprocalAmount, Temperature, Velocity,
};
use crate::math::Quantity;

/// The Avogadro constant.
pub const AVOGADRO_CONSTANT: Quantity<ReciprocalAmount> =
    ReciprocalAmount::per_mole(6.022_140_76e23);

/// The Boltzmann constant.
pub const BOLTZMANN_CONSTANT: Quantity<Entropy> = Entropy::joules_per_kelvin(1.380_649e-23);

/// The molar gas constant.
pub const GAS_CONSTANT: Quantity<MolarEntropy> =
    MolarEntropy::joules_per_mole_kelvin(8.314_462_618_153_24);

/// The elementary charge.
pub const ELEMENTARY_CHARGE: Quantity<Charge> = Charge::coulombs(1.602_176_634e-19);

/// The vacuum permeability.
pub const VACUUM_PERMEABILITY: Quantity<Permeability> =
    Permeability::henries_per_meter(1.256_637_062_12e-6);

/// The vacuum permittivity.
pub const VACUUM_PERMITTIVITY: Quantity<Permittivity> =
    Permittivity::farads_per_meter(8.854_187_812_8e-12);

/// The speed of light in vacuum.
pub const LIGHT_SPEED: Quantity<Velocity> = Velocity::meters_per_second(2.997_924_58e8);

/// The Planck constant.
pub const PLANCK_CONSTANT: Quantity<Action> = Action::joule_seconds(6.626_070_15e-34);

/// The standard acceleration of gravity.
pub const STANDARD_GRAVITY: Quantity<Acceleration> = Acceleration::standard_gravities(1.0);

/// Standard room temperature.
pub const ROOM_TEMPERATURE: Quantity<Temperature> = Temperature::celsius(20.0);

/// The Stefan-Boltzmann constant.
pub const STEFAN_BOLTZMANN_CONSTANT: Quantity<PowerPerAreaTemperatureFourth> =
    PowerPerAreaTemperatureFourth::watts_per_square_meter_kelvin_fourth(5.670_374_419e-8);
