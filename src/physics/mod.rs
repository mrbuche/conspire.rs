//! Physics models.

#[cfg(test)]
mod test;

/// Molecular physics models.
pub mod molecular;

use crate::{
    math::Quantity,
    units::{Action, Charge, Entropy, MolarEntropy, ReciprocalAmount, Temperature, Velocity},
};

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

/// The speed of light in vacuum.
pub const LIGHT_SPEED: Quantity<Velocity> = Velocity::meters_per_second(2.997_924_58e8);

/// The Planck constant.
pub const PLANCK_CONSTANT: Quantity<Action> = Action::joule_seconds(6.626_070_15e-34);

/// Standard room temperature.
pub const ROOM_TEMPERATURE: Quantity<Temperature> = Temperature::celsius(20.0);
