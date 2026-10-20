use super::Linear;
use crate::{
    constitutive::diffusion::surface::{SurfaceFlux, test::test_surface},
    math::{
        Quantity,
        assert::{Assert, AssertionError},
    },
    units::{Power, PowerPerArea, PowerPerAreaTemperature, Temperature},
};

fn model() -> Linear<Power> {
    Linear {
        coefficient: Quantity::<PowerPerAreaTemperature>::new(4.0),
        ambient: Quantity::<Temperature>::new(10.0),
    }
}

test_surface!(battery, model(), [3.0, 10.0, 14.0]);

#[test]
fn the_flux_is_the_coefficient_times_the_difference_from_the_ambient() -> Result<(), AssertionError>
{
    [(13.0, 12.0), (10.0, 0.0), (7.0, -12.0)]
        .into_iter()
        .try_for_each(|(field, flux)| {
            Assert::default().eq_within_tols(
                model().flux(Quantity::<Temperature>::new(field)),
                &Quantity::<PowerPerArea>::new(flux),
            )
        })
}

#[test]
fn the_tangent_is_the_coefficient() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        model().tangent(Quantity::<Temperature>::new(3.0)),
        &Quantity::<PowerPerAreaTemperature>::new(4.0),
    )
}
