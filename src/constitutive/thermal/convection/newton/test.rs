use super::Newton;
use crate::{
    constitutive::diffusion::surface::{SurfaceFlux, SurfacePotential},
    math::Quantity,
    units::{Area, Power, PowerPerAreaTemperature, Temperature},
};

fn law() -> Newton<Power> {
    Newton {
        coefficient: Quantity::<PowerPerAreaTemperature>::new(4.0),
        ambient: Quantity::<Temperature>::new(10.0),
    }
}

#[test]
fn the_flux_is_the_coefficient_times_the_difference_from_the_ambient() {
    assert_eq!(law().flux(Quantity::<Temperature>::new(13.0)).value(), 12.0);
    assert_eq!(law().flux(Quantity::<Temperature>::new(10.0)).value(), 0.0);
    assert_eq!(law().flux(Quantity::<Temperature>::new(7.0)).value(), -12.0)
}

#[test]
fn the_tangent_is_the_coefficient() {
    assert_eq!(
        law().tangent(Quantity::<Temperature>::new(3.0)).value(),
        4.0
    )
}

#[test]
fn the_flux_is_the_derivative_of_the_potential_per_area() {
    let (area, step) = (Quantity::<Area>::new(1.5), 1e-6);
    [3.0, 10.0, 14.0].into_iter().for_each(|field| {
        let potential = |field: f64| {
            law()
                .potential(Quantity::<Temperature>::new(field), area)
                .value()
        };
        let slope = (potential(field + step) - potential(field - step)) / (2.0 * step);
        let flux = law().flux(Quantity::<Temperature>::new(field)).value() * area.value();
        assert!(
            (slope - flux).abs() < 1e-6 * (1.0 + flux.abs()),
            "{slope} != {flux}"
        )
    })
}
