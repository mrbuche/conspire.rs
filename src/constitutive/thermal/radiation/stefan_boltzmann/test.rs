use super::StefanBoltzmann;
use crate::{
    constitutive::diffusion::surface::{SurfaceFlux, test::test_surface},
    math::{
        Quantity,
        assert::{Assert, AssertionError},
    },
    units::{PowerPerArea, STEFAN_BOLTZMANN_CONSTANT, Temperature},
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
fn the_flux_is_the_difference_of_the_fourth_powers_times_the_constant_and_the_emissivity()
-> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        model().flux(kelvin(400.0)),
        &Quantity::<PowerPerArea>::new(
            STEFAN_BOLTZMANN_CONSTANT.value() * 0.8 * (400.0_f64.powi(4) - 300.0_f64.powi(4)),
        ),
    )?;
    Assert::default().eq_within_tols(
        model().flux(kelvin(300.0)),
        &Quantity::<PowerPerArea>::new(0.0),
    )?;
    Assert::non_negative(&-model().flux(kelvin(200.0)))
}
