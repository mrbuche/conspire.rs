//! Physics models.

/// Molecular physics models.
pub mod molecular;

use crate::{
    math::Quantity,
    units::{Entropy, Temperature},
};

/// The Boltzmann constant.
pub const BOLTZMANN_CONSTANT: Quantity<Entropy> = Entropy::joules_per_kelvin(1.380_649e-23);

/// Standard room temperature.
pub const ROOM_TEMPERATURE: Quantity<Temperature> = Temperature::celsius(20.0);
