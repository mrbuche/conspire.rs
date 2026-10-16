use crate::{math::Quantity, units::ElectricalConductivity};

pub const ELECTRICAL_CONDUCTIVITY: Quantity<ElectricalConductivity> =
    ElectricalConductivity::siemens_per_meter(3.0);
