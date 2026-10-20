use super::PowerLaw;
use crate::{
    constitutive::diffusion::surface::{SurfaceFlux, test::test_surface},
    math::Quantity,
    units::{Power, PowerPerAreaTemperature, Temperature},
};

fn law(exponent: f64) -> PowerLaw<Power> {
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

test_surface!(linear, law(1.0), [3.0, 9.5, 10.0, 10.5, 14.0, 30.0]);
test_surface!(laminar, law(1.25), [3.0, 9.5, 10.0, 10.5, 14.0, 30.0]);
test_surface!(
    turbulent,
    law(4.0 / 3.0),
    [3.0, 9.5, 10.0, 10.5, 14.0, 30.0]
);
test_surface!(quadratic, law(2.0), [3.0, 9.5, 10.0, 10.5, 14.0, 30.0]);

#[test]
fn the_flux_is_the_coefficient_times_the_difference_at_the_reference_difference() {
    [1.0, 1.25, 4.0 / 3.0, 2.0]
        .into_iter()
        .for_each(|exponent| {
            assert!((law(exponent).flux(kelvin(12.0)).value() - 8.0).abs() < 1e-12);
            assert!((law(exponent).flux(kelvin(8.0)).value() + 8.0).abs() < 1e-12);
            assert_eq!(law(exponent).flux(kelvin(10.0)).value(), 0.0)
        })
}

#[test]
fn the_flux_grows_as_the_difference_to_the_exponent() {
    let exponent = 1.25;
    let found = law(exponent).flux(kelvin(26.0)).value();
    let expected = 4.0 * 16.0 * (16.0_f64 / 2.0).powf(exponent - 1.0);
    assert!((found - expected).abs() < 1e-12 * expected)
}

#[test]
fn an_exponent_of_one_is_linear() {
    [3.0, 10.0, 14.0].into_iter().for_each(|field| {
        assert!((law(1.0).flux(kelvin(field)).value() - 4.0 * (field - 10.0)).abs() < 1e-12);
        assert!((law(1.0).tangent(kelvin(field)).value() - 4.0).abs() < 1e-12)
    })
}
