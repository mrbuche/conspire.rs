//! Solid-thermal constitutive models.

pub(super) mod thermoelastic_thermal_conduction;
pub(super) mod thermohyperelastic_thermal_conduction;

use super::{
    super::{solid::Solid, thermal::Thermal},
    Multiphysics,
};

/// Required methods for solid-thermal constitutive models.
pub trait SolidThermal
where
    Self: Multiphysics,
{
    /// The solid constitutive model.
    type Solid: Solid;
    /// The thermal constitutive model.
    type Thermal: Thermal;
    /// Returns a reference to the solid constitutive model.
    fn solid_constitutive_model(&self) -> &Self::Solid;
    /// Returns a reference to the thermal constitutive model.
    fn thermal_constitutive_model(&self) -> &Self::Thermal;
}
