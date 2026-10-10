//! Solid-thermal constitutive models.

#[cfg(test)]
pub mod test;

pub(super) mod thermoelastic_thermal_conduction;
pub(super) mod thermohyperelastic_thermal_conduction;

use super::{
    super::{solid::Solid, thermal::Thermal},
    Multiphysics,
};
use crate::{math::Quantity, units::TimePerTemperature};

/// Required methods for solid-thermal constitutive models.
pub trait SolidThermal
where
    Self: Multiphysics + Solid + Thermal,
{
    /// The solid constitutive model.
    type Solid: Solid;
    /// The thermal constitutive model.
    type Thermal: Thermal;
    /// Returns a reference to the solid constitutive model.
    fn solid_constitutive_model(&self) -> &Self::Solid;
    /// Returns a reference to the thermal constitutive model.
    fn thermal_constitutive_model(&self) -> &Self::Thermal;
    /// Returns the solid-thermal coupling parameter.
    fn coupling_parameter(&self) -> Quantity<TimePerTemperature>;
}
