use crate::{math::Quantity, units::TimePerTemperature};

pub const COUPLING_PARAMETER: Quantity<TimePerTemperature> =
    TimePerTemperature::seconds_per_kelvin(2.0);
