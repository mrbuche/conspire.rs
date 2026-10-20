#[cfg(test)]
mod test;

use super::{SurfaceField, SurfaceFlux, SurfaceFluxPerArea, SurfacePotential, SurfaceTangent};
use crate::{constitutive::diffusion::Diffusive, math::Quantity, units::Area};
use std::ops::Mul;

/// A linear law, where the flux is proportional to the difference between the field and its
/// ambient value.
#[derive(Clone, Debug)]
pub struct Linear<L>
where
    L: Diffusive,
{
    /// The coefficient of the flux.
    pub coefficient: Quantity<L::TangentPerLength>,
    /// The ambient value of the field.
    pub ambient: Quantity<L::Field>,
}

impl<L> SurfaceFlux for Linear<L>
where
    L: Diffusive,
    Quantity<L::TangentPerLength>: Mul<Quantity<L::Field>, Output = Quantity<L::Flux>>,
{
    type Balance = L;
    fn flux(&self, field: SurfaceField<Self>) -> SurfaceFluxPerArea<Self> {
        self.coefficient * (field - self.ambient)
    }
    fn tangent(&self, _: SurfaceField<Self>) -> SurfaceTangent<Self> {
        self.coefficient
    }
}

impl<L> SurfacePotential for Linear<L>
where
    L: Diffusive,
    Quantity<L::TangentPerLength>: Mul<Quantity<L::Field>, Output = Quantity<L::Flux>>
        + Mul<Quantity<Area>, Output = Quantity<L::Stiffness>>,
    Quantity<L::Stiffness>: Mul<Quantity<L::Field>, Output = Quantity<L>>,
    Quantity<L::Field>: Mul<Quantity<L>, Output = Quantity<L::Energy>>,
{
    fn potential(&self, field: SurfaceField<Self>, area: Quantity<Area>) -> Quantity<L::Energy> {
        let difference = field - self.ambient;
        difference * ((self.coefficient * area) * difference) * 0.5
    }
}
