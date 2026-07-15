use super::{super::test::THERMAL_CONDUCTIVITY, Fourier, TemperatureGradient, ThermalConduction};
use crate::math::assert::Assert;
use crate::{
    math::{Scalar, Tensor},
    mechanics::test::get_temperature_gradient,
};

#[test]
fn size() {
    assert_eq!(
        std::mem::size_of::<Fourier>(),
        std::mem::size_of::<Scalar>()
    )
}

#[test]
fn thermal_conductivity() -> Result<(), crate::math::assert::AssertionError> {
    let model = Fourier {
        thermal_conductivity: THERMAL_CONDUCTIVITY,
    };
    model
        .heat_flux(&get_temperature_gradient())?
        .iter()
        .zip((get_temperature_gradient() / -model.thermal_conductivity()).iter())
        .try_for_each(|(heat_flux_i, entry_i)| Assert::eq(heat_flux_i, entry_i))
}

#[test]
fn zero() -> Result<(), crate::math::assert::AssertionError> {
    Fourier {
        thermal_conductivity: THERMAL_CONDUCTIVITY,
    }
    .heat_flux(&TemperatureGradient::from([0.0, 0.0, 0.0]))?
    .iter()
    .try_for_each(|heat_flux_i| Assert::eq(heat_flux_i, &0.0))
}
