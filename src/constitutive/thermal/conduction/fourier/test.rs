use super::{
    super::test::FOURIERPARAMETERS, Constitutive, Fourier, TemperatureGradient, ThermalConduction,
};
use crate::{
    math::{Tensor, TensorArray},
    mechanics::test::get_temperature_gradient,
};

fn get_constitutive_model<'a>() -> Fourier<'a> {
    Fourier::new(FOURIERPARAMETERS)
}

#[test]
fn get_thermal_conductivity() {
    assert_eq!(
        &FOURIERPARAMETERS[0],
        get_constitutive_model().get_thermal_conductivity()
    )
}

#[test]
fn size() {
    assert_eq!(
        std::mem::size_of::<Fourier>(),
        std::mem::size_of::<crate::constitutive::Parameters>()
    )
}

#[test]
fn thermal_conductivity() {
    let model = get_constitutive_model();
    model
        .calculate_heat_flux(&get_temperature_gradient())
        .iter()
        .zip((get_temperature_gradient() / -model.get_thermal_conductivity()).iter())
        .for_each(|(heat_flux_i, entry_i)| assert_eq!(heat_flux_i, entry_i))
}

#[test]
fn zero() {
    get_constitutive_model()
        .calculate_heat_flux(&TemperatureGradient::new([0.0, 0.0, 0.0]))
        .iter()
        .for_each(|heat_flux_i| assert_eq!(heat_flux_i, &0.0))
}
