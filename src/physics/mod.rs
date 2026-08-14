//! Physics library.

/// Molecular physics models.
pub mod molecular;

use crate::math::{Quantity, Scalar, unit::Entropy};

/// The Boltzmann constant in units of J/(mol⋅K).
pub const BOLTZMANN_CONSTANT: Quantity<Entropy> = Quantity::new(8.314_462_618_153_24);

/// Standard room temperature in units of K.
pub const ROOM_TEMPERATURE: Scalar = 273.15;
