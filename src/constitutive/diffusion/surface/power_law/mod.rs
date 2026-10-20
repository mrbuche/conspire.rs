#[cfg(test)]
mod test;

use super::{SurfaceField, SurfaceFlux, SurfaceFluxPerArea, SurfacePotential, SurfaceTangent};
use crate::{
    constitutive::diffusion::Diffusive,
    math::{Quantity, Scalar},
    units::{Area, Dimensionless},
};
use std::ops::{Div, Mul};

/// A power law, where the flux is the coefficient times the difference between the field and its
/// ambient value, times that difference over a reference difference, in magnitude, to a power
/// less one.
///
/// The exponent is at least one, and the coefficient is that of the flux at the reference
/// difference. An exponent of one is a linear law.
#[derive(Clone, Debug)]
pub struct PowerLaw<L>
where
    L: Diffusive,
{
    /// The coefficient of the flux at the reference difference.
    pub coefficient: Quantity<L::TangentPerLength>,
    /// The ambient value of the field.
    pub ambient: Quantity<L::Field>,
    /// The difference from the ambient value at which the coefficient holds.
    pub reference: Quantity<L::Field>,
    /// The exponent of the flux.
    pub exponent: Scalar,
}

impl<L> PowerLaw<L>
where
    L: Diffusive,
    Quantity<L::Field>: Div<Quantity<L::Field>, Output = Quantity<Dimensionless>>,
{
    fn scale(&self, difference: Quantity<L::Field>) -> Scalar {
        (difference / self.reference)
            .value()
            .abs()
            .powf(self.exponent - 1.0)
    }
}

impl<L> SurfaceFlux for PowerLaw<L>
where
    L: Diffusive,
    Quantity<L::TangentPerLength>: Mul<Quantity<L::Field>, Output = Quantity<L::Flux>>,
    Quantity<L::Field>: Div<Quantity<L::Field>, Output = Quantity<Dimensionless>>,
{
    type Balance = L;
    fn flux(&self, field: SurfaceField<Self>) -> SurfaceFluxPerArea<Self> {
        let difference = field - self.ambient;
        self.coefficient * difference * self.scale(difference)
    }
    fn tangent(&self, field: SurfaceField<Self>) -> SurfaceTangent<Self> {
        self.exponent * self.scale(field - self.ambient) * self.coefficient
    }
}

impl<L> SurfacePotential for PowerLaw<L>
where
    L: Diffusive,
    Quantity<L::TangentPerLength>: Mul<Quantity<L::Field>, Output = Quantity<L::Flux>>
        + Mul<Quantity<Area>, Output = Quantity<L::Stiffness>>,
    Quantity<L::Stiffness>: Mul<Quantity<L::Field>, Output = Quantity<L>>,
    Quantity<L::Field>: Mul<Quantity<L>, Output = Quantity<L::Energy>>
        + Div<Quantity<L::Field>, Output = Quantity<Dimensionless>>,
{
    fn potential(&self, field: SurfaceField<Self>, area: Quantity<Area>) -> Quantity<L::Energy> {
        let difference = field - self.ambient;
        difference
            * ((self.coefficient * area) * difference)
            * (self.scale(difference) / (self.exponent + 1.0))
    }
}
