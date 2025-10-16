use super::{super::test::THERMAL_CONDUCTIVITY, Fourier, TemperatureGradient, ThermalConduction};
use crate::{
    math::{Scalar, Tensor, TensorArray},
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
fn thermal_conductivity() {
    let model = Fourier {
        thermal_conductivity: THERMAL_CONDUCTIVITY,
    };
    model
        .heat_flux(&get_temperature_gradient())
        .iter()
        .zip((get_temperature_gradient() / -model.thermal_conductivity()).iter())
        .for_each(|(heat_flux_i, entry_i)| assert_eq!(heat_flux_i, entry_i))
}

#[test]
fn zero() {
    Fourier {
        thermal_conductivity: THERMAL_CONDUCTIVITY,
    }
    .heat_flux(&TemperatureGradient::new([0.0, 0.0, 0.0]))
    .iter()
    .for_each(|heat_flux_i| assert_eq!(heat_flux_i, &0.0))
}
