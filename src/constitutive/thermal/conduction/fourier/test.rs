use super::{
    super::test::FOURIERPARAMETERS, Constitutive, Fourier, TemperatureGradient, ThermalConduction,
};
use crate::{
    math::{Tensor, TensorArray},
    mechanics::{Scalar, test::get_temperature_gradient},
};

type FourierType<'a> = Fourier<&'a [Scalar; 1]>;

fn get_constitutive_model<'a>() -> FourierType<'a> {
    Fourier::new(FOURIERPARAMETERS)
}

#[test]
fn size() {
    assert_eq!(
        std::mem::size_of::<FourierType>(),
        std::mem::size_of::<&[Scalar; 1]>()
    )
}

#[test]
fn thermal_conductivity() {
    let model = get_constitutive_model();
    assert_eq!(&FOURIERPARAMETERS[0], model.thermal_conductivity());
    model
        .heat_flux(&get_temperature_gradient())
        .iter()
        .zip((get_temperature_gradient() / -model.thermal_conductivity()).iter())
        .for_each(|(heat_flux_i, entry_i)| assert_eq!(heat_flux_i, entry_i))
}

#[test]
fn zero() {
    get_constitutive_model()
        .heat_flux(&TemperatureGradient::new([0.0, 0.0, 0.0]))
        .iter()
        .for_each(|heat_flux_i| assert_eq!(heat_flux_i, &0.0))
}
