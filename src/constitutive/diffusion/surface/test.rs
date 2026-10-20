macro_rules! test_surface_flux {
    ($name: ident, $law: expr, $fields: expr) => {
        mod $name {
            use super::*;
            #[test]
            fn the_tangent_is_the_derivative_of_the_flux() {
                $fields.into_iter().for_each(|field: f64| {
                    let step = 1e-6 * (1.0 + field.abs());
                    let flux = |field: f64| {
                        crate::constitutive::diffusion::surface::SurfaceFlux::flux(
                            &$law,
                            crate::math::Quantity::new(field),
                        )
                        .value()
                    };
                    let slope = (flux(field + step) - flux(field - step)) / (2.0 * step);
                    let tangent = crate::constitutive::diffusion::surface::SurfaceFlux::tangent(
                        &$law,
                        crate::math::Quantity::new(field),
                    )
                    .value();
                    assert!(
                        (slope - tangent).abs() < 1e-6 * tangent.abs() + 1e-9,
                        "{field}: {slope} != {tangent}"
                    )
                })
            }
            #[test]
            fn the_tangent_is_positive() {
                $fields.into_iter().for_each(|field: f64| {
                    let tangent = crate::constitutive::diffusion::surface::SurfaceFlux::tangent(
                        &$law,
                        crate::math::Quantity::new(field),
                    )
                    .value();
                    assert!(tangent >= 0.0, "{field}: {tangent}")
                })
            }
            #[test]
            fn the_flux_increases_with_the_field() {
                let mut fields = $fields.into_iter().collect::<Vec<f64>>();
                fields.sort_by(f64::total_cmp);
                let fluxes = fields
                    .into_iter()
                    .map(|field| {
                        crate::constitutive::diffusion::surface::SurfaceFlux::flux(
                            &$law,
                            crate::math::Quantity::new(field),
                        )
                        .value()
                    })
                    .collect::<Vec<f64>>();
                assert!(
                    fluxes.windows(2).all(|pair| pair[0] <= pair[1]),
                    "{fluxes:?}"
                )
            }
        }
    };
}
pub(crate) use test_surface_flux;

macro_rules! test_surface_potential {
    ($name: ident, $law: expr, $fields: expr) => {
        mod $name {
            use super::*;
            #[test]
            fn the_flux_is_the_derivative_of_the_potential_per_area() {
                let area = crate::math::Quantity::<crate::units::Area>::new(1.5);
                $fields.into_iter().for_each(|field: f64| {
                    let step = 1e-6 * (1.0 + field.abs());
                    let potential = |field: f64| {
                        crate::constitutive::diffusion::surface::SurfacePotential::potential(
                            &$law,
                            crate::math::Quantity::new(field),
                            area,
                        )
                        .value()
                    };
                    let slope = (potential(field + step) - potential(field - step)) / (2.0 * step);
                    let flux = crate::constitutive::diffusion::surface::SurfaceFlux::flux(
                        &$law,
                        crate::math::Quantity::new(field),
                    )
                    .value()
                        * area.value();
                    assert!(
                        (slope - flux).abs() < 1e-6 * flux.abs() + 1e-9,
                        "{field}: {slope} != {flux}"
                    )
                })
            }
            #[test]
            fn the_potential_is_the_minimum_where_the_flux_vanishes() {
                let area = crate::math::Quantity::<crate::units::Area>::new(1.5);
                $fields.into_iter().for_each(|field: f64| {
                    let flux = crate::constitutive::diffusion::surface::SurfaceFlux::flux(
                        &$law,
                        crate::math::Quantity::new(field),
                    )
                    .value();
                    if flux == 0.0 {
                        let potential = |field: f64| {
                            crate::constitutive::diffusion::surface::SurfacePotential::potential(
                                &$law,
                                crate::math::Quantity::new(field),
                                area,
                            )
                            .value()
                        };
                        let nearby = [-1e-2, 1e-2].map(|step| potential(field + step));
                        assert!(nearby.into_iter().all(|value| value >= potential(field)))
                    }
                })
            }
        }
    };
}
pub(crate) use test_surface_potential;

macro_rules! test_surface {
    ($name: ident, $law: expr, $fields: expr) => {
        mod $name {
            use super::*;
            crate::constitutive::diffusion::surface::test::test_surface_flux!(flux, $law, $fields);
            crate::constitutive::diffusion::surface::test::test_surface_potential!(
                potential, $law, $fields
            );
        }
    };
}
pub(crate) use test_surface;
