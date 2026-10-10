//! Thermoelastic-thermal conduction constitutive models.

pub(crate) mod canonical;

use crate::constitutive::{
    multiphysics::SolidThermal, solid::thermoelastic::Thermoelastic,
    thermal::conduction::ThermalConduction,
};

pub use canonical::Canonical;

/// Required methods for thermoelastic-thermal conduction constitutive models.
pub trait ThermoelasticThermalConduction
where
    Self: SolidThermal<Solid: Thermoelastic, Thermal: ThermalConduction>,
{
}

impl<T> ThermoelasticThermalConduction for T where
    T: SolidThermal<Solid: Thermoelastic, Thermal: ThermalConduction>
{
}
