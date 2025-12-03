//! Solid-thermal constitutive models.

pub mod thermoelastic_thermal_conduction;
pub mod thermohyperelastic_thermal_conduction;

use super::{
    super::{solid::Solid, thermal::Thermal},
    Multiphysics,
};

/// Required methods for solid-thermal constitutive models.
pub trait SolidThermal<C1, C2>
where
    C1: Solid,
    C2: Thermal,
    Self: Multiphysics,
{
    /// Returns a reference to the solid constitutive model.
    fn solid_constitutive_model(&self) -> &C1;
    /// Returns a reference to the thermal constitutive model.
    fn thermal_constitutive_model(&self) -> &C2;
}
