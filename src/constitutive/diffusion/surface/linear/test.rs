use super::Linear;
use crate::{
    constitutive::diffusion::surface::{SurfaceFlux, test::test_surface},
    math::Quantity,
    units::{Power, PowerPerAreaTemperature, Temperature},
};

fn model() -> Linear<Power> {
    Linear {
        coefficient: Quantity::<PowerPerAreaTemperature>::new(4.0),
        ambient: Quantity::<Temperature>::new(10.0),
    }
}

test_surface!(battery, model(), [3.0, 10.0, 14.0]);

#[test]
fn the_flux_is_the_coefficient_times_the_difference_from_the_ambient() {
    assert_eq!(
        model().flux(Quantity::<Temperature>::new(13.0)).value(),
        12.0
    );
    assert_eq!(
        model().flux(Quantity::<Temperature>::new(10.0)).value(),
        0.0
    );
    assert_eq!(
        model().flux(Quantity::<Temperature>::new(7.0)).value(),
        -12.0
    )
}

#[test]
fn the_tangent_is_the_coefficient() {
    assert_eq!(
        model().tangent(Quantity::<Temperature>::new(3.0)).value(),
        4.0
    )
}
