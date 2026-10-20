use super::PowerLaw;
use crate::{
    constitutive::diffusion::surface::{SurfaceFlux, test::test_surface},
    math::{
        Quantity,
        assert::{Assert, AssertionError},
    },
    units::{Power, PowerPerArea, PowerPerAreaTemperature, Temperature},
};

fn model(exponent: f64) -> PowerLaw<Power> {
    PowerLaw {
        coefficient: Quantity::<PowerPerAreaTemperature>::new(4.0),
        ambient: Quantity::<Temperature>::new(10.0),
        reference: Quantity::<Temperature>::new(2.0),
        exponent,
    }
}

fn kelvin(value: f64) -> Quantity<Temperature> {
    Quantity::<Temperature>::new(value)
}

fn flux(value: f64) -> Quantity<PowerPerArea> {
    Quantity::<PowerPerArea>::new(value)
}

test_surface!(linear, model(1.0), [3.0, 9.5, 10.0, 10.5, 14.0, 30.0]);
test_surface!(laminar, model(1.25), [3.0, 9.5, 10.0, 10.5, 14.0, 30.0]);
test_surface!(
    turbulent,
    model(4.0 / 3.0),
    [3.0, 9.5, 10.0, 10.5, 14.0, 30.0]
);
test_surface!(quadratic, model(2.0), [3.0, 9.5, 10.0, 10.5, 14.0, 30.0]);

#[test]
fn the_flux_is_the_coefficient_times_the_difference_at_the_reference_difference()
-> Result<(), AssertionError> {
    [1.0, 1.25, 4.0 / 3.0, 2.0]
        .into_iter()
        .try_for_each(|exponent| {
            Assert::default().eq_within_tols(model(exponent).flux(kelvin(12.0)), &flux(8.0))?;
            Assert::default().eq_within_tols(model(exponent).flux(kelvin(8.0)), &flux(-8.0))?;
            Assert::default().eq_within_tols(model(exponent).flux(kelvin(10.0)), &flux(0.0))
        })
}

#[test]
fn the_flux_grows_as_the_difference_to_the_exponent() -> Result<(), AssertionError> {
    let exponent = 1.25;
    Assert::default().eq_within_tols(
        model(exponent).flux(kelvin(26.0)),
        &flux(4.0 * 16.0 * (16.0_f64 / 2.0).powf(exponent - 1.0)),
    )
}

#[test]
fn an_exponent_of_one_is_linear() -> Result<(), AssertionError> {
    [3.0, 10.0, 14.0].into_iter().try_for_each(|field| {
        Assert::default()
            .eq_within_tols(model(1.0).flux(kelvin(field)), &flux(4.0 * (field - 10.0)))?;
        Assert::default().eq_within_tols(
            model(1.0).tangent(kelvin(field)),
            &Quantity::<PowerPerAreaTemperature>::new(4.0),
        )
    })
}
