use super::StefanBoltzmann;
use crate::{
    constitutive::diffusion::surface::{SurfaceFlux, test::test_surface},
    math::Quantity,
    units::{STEFAN_BOLTZMANN_CONSTANT, Temperature},
};

fn model() -> StefanBoltzmann {
    StefanBoltzmann {
        emissivity: 0.8,
        ambient: Quantity::<Temperature>::new(300.0),
    }
}

fn kelvin(value: f64) -> Quantity<Temperature> {
    Quantity::<Temperature>::new(value)
}

test_surface!(battery, model(), [200.0, 300.0, 450.0]);

#[test]
fn the_flux_is_the_difference_of_the_fourth_powers_times_the_constant_and_the_emissivity() {
    let found = model().flux(kelvin(400.0)).value();
    let expected =
        STEFAN_BOLTZMANN_CONSTANT.value() * 0.8 * (400.0_f64.powi(4) - 300.0_f64.powi(4));
    assert!((found - expected).abs() < 1e-12 * expected);
    assert_eq!(model().flux(kelvin(300.0)).value(), 0.0);
    assert!(model().flux(kelvin(200.0)).value() < 0.0)
}
