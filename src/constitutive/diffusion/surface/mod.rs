//! Models of the flux through a surface.

#[cfg(test)]
pub mod test;

pub mod linear;
pub mod power_law;

pub use linear::Linear;
pub use power_law::PowerLaw;

use crate::{constitutive::diffusion::Diffusive, math::Quantity, units::Area};

/// The field of the balance law of a surface flux.
pub type SurfaceField<S> = Quantity<<<S as SurfaceFlux>::Balance as Diffusive>::Field>;

/// The flux of a surface flux, per unit area.
pub type SurfaceFluxPerArea<S> = Quantity<<<S as SurfaceFlux>::Balance as Diffusive>::Flux>;

/// The tangent of a surface flux, per unit area.
pub type SurfaceTangent<S> = Quantity<<<S as SurfaceFlux>::Balance as Diffusive>::TangentPerLength>;

/// Required methods for models of the flux out of a body through a surface, as a function of the
/// field on it.
pub trait SurfaceFlux {
    /// The balance law of the flux.
    type Balance: Diffusive;
    /// The flux out of the body per unit area.
    fn flux(&self, field: SurfaceField<Self>) -> SurfaceFluxPerArea<Self>;
    /// The derivative of the flux per unit area with respect to the field.
    fn tangent(&self, field: SurfaceField<Self>) -> SurfaceTangent<Self>;
}

/// Required methods for models of the flux through a surface whose flux is the derivative of a
/// potential with respect to the field.
pub trait SurfacePotential
where
    Self: SurfaceFlux,
{
    /// The potential of a part of the surface with an area.
    fn potential(
        &self,
        field: SurfaceField<Self>,
        area: Quantity<Area>,
    ) -> Quantity<<Self::Balance as Diffusive>::Energy>;
}
