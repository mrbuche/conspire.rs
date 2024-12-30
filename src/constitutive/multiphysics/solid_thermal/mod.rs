//! Solid-thermal constitutive models.

pub mod thermoelastic_thermal_conduction;
pub mod thermohyperelastic_thermal_conduction;

use super::*;

/// Required methods for solid-thermal constitutive models.
pub trait SolidThermal<'a, C1, C2>
where
    C1: Solid<'a>,
    C2: Thermal<'a>,
    Self: Multiphysics<'a>,
{
    /// Constructs and returns a new solid-thermal constitutive model.
    fn construct(solid_constitutive_model: C1, thermal_constitutive_model: C2) -> Self;
    /// Returns a reference to the solid constitutive model.
    fn get_solid_constitutive_model(&self) -> &C1;
    /// Returns a reference to the thermal constitutive model.
    fn get_thermal_constitutive_model(&self) -> &C2;
}
