use crate::{math::Quantity, units::PowerPerLengthTemperature};

pub const THERMAL_CONDUCTIVITY: Quantity<PowerPerLengthTemperature> =
    PowerPerLengthTemperature::watts_per_meter_kelvin(1.0);
