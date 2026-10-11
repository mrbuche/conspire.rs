//! Thermohyperelastic-thermal conduction constitutive models.

mod canonical;

use crate::constitutive::{
    multiphysics::SolidThermal, solid::thermohyperelastic::Thermohyperelastic,
    thermal::conduction::ThermalConduction,
};

/// Required methods for thermohyperelastic-thermal conduction constitutive models.
pub trait ThermohyperelasticThermalConduction
where
    Self: SolidThermal<Solid: Thermohyperelastic, Thermal: ThermalConduction>,
{
}

impl<T> ThermohyperelasticThermalConduction for T where
    T: SolidThermal<Solid: Thermohyperelastic, Thermal: ThermalConduction>
{
}
