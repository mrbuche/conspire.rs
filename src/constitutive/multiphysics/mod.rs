//! Multiphysics constitutive models.

mod solid_thermal;

pub use solid_thermal::{
    SolidThermal, thermoelastic_thermal_conduction::ThermoelasticThermalConduction,
    thermohyperelastic_thermal_conduction::ThermohyperelasticThermalConduction,
};

/// Required methods for multiphysics constitutive models.
pub trait Multiphysics {}
