use super::StefanBoltzmann;
use crate::{
    constitutive::diffusion::surface::{SurfaceFlux, SurfacePotential},
    math::Quantity,
    units::{Area, STEFAN_BOLTZMANN_CONSTANT, Temperature},
};

fn law() -> StefanBoltzmann {
    StefanBoltzmann {
        emissivity: 0.8,
        ambient: Quantity::<Temperature>::new(300.0),
    }
}

fn kelvin(value: f64) -> Quantity<Temperature> {
    Quantity::<Temperature>::new(value)
}

#[test]
fn the_flux_is_the_difference_of_the_fourth_powers_times_the_constant_and_the_emissivity() {
    let found = law().flux(kelvin(400.0)).value();
    let expected =
        STEFAN_BOLTZMANN_CONSTANT.value() * 0.8 * (400.0_f64.powi(4) - 300.0_f64.powi(4));
    assert!((found - expected).abs() < 1e-12 * expected);
    assert_eq!(law().flux(kelvin(300.0)).value(), 0.0);
    assert!(law().flux(kelvin(200.0)).value() < 0.0)
}

#[test]
fn the_tangent_is_the_derivative_of_the_flux() {
    let step = 1e-4;
    [200.0, 300.0, 450.0].into_iter().for_each(|field| {
        let flux = |field: f64| law().flux(kelvin(field)).value();
        let slope = (flux(field + step) - flux(field - step)) / (2.0 * step);
        let tangent = law().tangent(kelvin(field)).value();
        assert!(
            (slope - tangent).abs() < 1e-6 * tangent,
            "{slope} != {tangent}"
        )
    })
}

#[test]
fn the_flux_is_the_derivative_of_the_potential_per_area() {
    let (area, step) = (Quantity::<Area>::new(1.5), 1e-4);
    [200.0, 300.0, 450.0].into_iter().for_each(|field| {
        let potential = |field: f64| law().potential(kelvin(field), area).value();
        let slope = (potential(field + step) - potential(field - step)) / (2.0 * step);
        let flux = law().flux(kelvin(field)).value() * area.value();
        assert!(
            (slope - flux).abs() < 1e-6 * (1.0 + flux.abs()),
            "{slope} != {flux}"
        )
    })
}
