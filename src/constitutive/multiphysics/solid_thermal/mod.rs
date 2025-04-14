//! Solid-thermal constitutive models.

pub mod thermoelastic_thermal_conduction;
// pub mod thermohyperelastic_thermal_conduction;

use crate::constitutive::Parameters;
use super::*;

/// Required methods for solid-thermal constitutive models.
pub trait SolidThermal<C1, C2, P1, P2>
where
    C1: Solid<P1>,
    C2: Thermal<P2>,
    // Self: Multiphysics<P>,
{
    /// Constructs and returns a new solid-thermal constitutive model.
    fn construct(solid_constitutive_model: C1, thermal_constitutive_model: C2) -> Self;
    /// Returns a reference to the solid constitutive model.
    fn solid_constitutive_model(&self) -> &C1;
    /// Returns a reference to the thermal constitutive model.
    fn thermal_constitutive_model(&self) -> &C2;
}
