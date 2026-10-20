macro_rules! test_surface_flux {
    ($name: ident, $model: expr, $fields: expr) => {
        crate::constitutive::diffusion::surface::test::test_surface_flux!(
            $name, $model, $fields, $fields
        );
    };
    ($name: ident, $model: expr, $fields: expr, $smooth: expr) => {
        mod $name {
            use super::*;
            #[test]
            fn the_tangent_is_the_derivative_of_the_flux() -> Result<(), AssertionError> {
                $smooth.into_iter().try_for_each(|field: f64| {
                    let (field, step) = (Quantity::new(field), perturbation(EPSILON));
                    let flux = |field| SurfaceFlux::flux(&$model, field);
                    let slope = (flux(field + step) - flux(field - step)) / (2.0 * step);
                    let tangent = SurfaceFlux::tangent(&$model, field);
                    Assert::default().eq_within_fd_tol(&slope, &tangent)
                })
            }
            #[test]
            fn the_tangent_is_not_negative() -> Result<(), AssertionError> {
                $fields.into_iter().try_for_each(|field: f64| {
                    Assert::non_negative(&SurfaceFlux::tangent(&$model, Quantity::new(field)))
                })
            }
            #[test]
            fn the_flux_does_not_decrease_with_the_field() {
                let mut fields = $fields.into_iter().collect::<Vec<f64>>();
                fields.sort_by(f64::total_cmp);
                let fluxes = fields
                    .into_iter()
                    .map(|field| SurfaceFlux::flux(&$model, Quantity::new(field)))
                    .collect::<Vec<_>>();
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
    ($name: ident, $model: expr, $fields: expr) => {
        mod $name {
            use super::*;
            #[test]
            fn the_flux_is_the_derivative_of_the_potential_per_area() -> Result<(), AssertionError>
            {
                let area = Quantity::<Area>::new(1.5);
                $fields.into_iter().try_for_each(|field: f64| {
                    let (field, step) = (Quantity::new(field), perturbation(EPSILON));
                    let potential = |field| SurfacePotential::potential(&$model, field, area);
                    let slope = (potential(field + step) - potential(field - step)) / (2.0 * step);
                    let flux = SurfaceFlux::flux(&$model, field) * area;
                    Assert::default().eq_within_fd_tol(&slope, &flux)
                })
            }
            #[test]
            fn the_potential_is_the_least_where_the_flux_vanishes() {
                let area = Quantity::<Area>::new(1.5);
                $fields.into_iter().for_each(|field: f64| {
                    let field = Quantity::new(field);
                    let flux = SurfaceFlux::flux(&$model, field);
                    if flux.value() == 0.0 {
                        let potential = |field| SurfacePotential::potential(&$model, field, area);
                        let step = perturbation(1e-2);
                        assert!(potential(field + step) >= potential(field));
                        assert!(potential(field - step) >= potential(field))
                    }
                })
            }
        }
    };
}
pub(crate) use test_surface_potential;

macro_rules! test_surface {
    ($name: ident, $model: expr, $fields: expr) => {
        crate::constitutive::diffusion::surface::test::test_surface!(
            $name, $model, $fields, $fields
        );
    };
    ($name: ident, $model: expr, $fields: expr, $smooth: expr) => {
        mod $name {
            use super::*;
            use crate::{
                EPSILON,
                constitutive::diffusion::surface::{SurfaceFlux, SurfacePotential},
                math::{
                    Quantity,
                    assert::{Assert, AssertionError, perturbation},
                },
                units::Area,
            };
            crate::constitutive::diffusion::surface::test::test_surface_flux!(
                flux, $model, $fields, $smooth
            );
            crate::constitutive::diffusion::surface::test::test_surface_potential!(
                potential, $model, $fields
            );
        }
    };
}
pub(crate) use test_surface;
