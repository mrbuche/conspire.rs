//! Multiphysics constitutive models.

mod solid_thermal;

pub use solid_thermal::{
    thermoelastic_thermal_conduction::ThermoelasticThermalConduction,
    thermohyperelastic_thermal_conduction::ThermohyperelasticThermalConduction, SolidThermal,
};

use super::{
    solid::{thermoelastic::Thermoelastic, thermohyperelastic::Thermohyperelastic, Solid},
    thermal::{conduction::ThermalConduction, Thermal},
    Constitutive, ConstitutiveError, Parameters,
};

/// Required methods for multiphysics constitutive models.
pub trait Multiphysics<'a>
where
    Self: Constitutive<'a>,
{
}
