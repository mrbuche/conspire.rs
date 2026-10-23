//! Irreversible cohesive constitutive models.

mod tvergaard_hutchinson;

pub use tvergaard_hutchinson::TvergaardHutchinson;

use crate::{
    constitutive::{
        ConstitutiveError,
        cohesive::{self, Cohesive, StiffnessCohesive, Stiffnesses},
    },
    math::{Quantity, integrate::StepError},
    mechanics::{Normal, Separation, Traction},
    units::{ForcePerLength, Length, Stress},
};

/// Required methods for irreversible cohesive constitutive models.
pub trait Damage
where
    Self: Cohesive,
{
    /// Calculates and returns the traction.
    fn traction(
        &self,
        separation: Separation,
        normal: Normal,
        state: Quantity,
    ) -> Result<Traction, ConstitutiveError> {
        cohesive::traction(
            separation,
            normal,
            |normal_separation, tangential_separation| {
                self.tractions(normal_separation, tangential_separation, state)
            },
        )
    }
    /// Calculates and returns the normal and tangential tractions.
    fn tractions(
        &self,
        normal_separation: Quantity<Length>,
        tangential_separation: Quantity<Length>,
        state: Quantity,
    ) -> Result<(Quantity<Stress>, Quantity<Stress>), ConstitutiveError>;
    /// Calculates and returns the tangent stiffness at a fixed state.
    fn stiffness(
        &self,
        separation: Separation,
        normal: Normal,
        state: Quantity,
    ) -> Result<StiffnessCohesive, ConstitutiveError> {
        cohesive::stiffness(
            separation,
            normal,
            |normal_separation, tangential_separation| {
                Ok((
                    self.tractions(normal_separation, tangential_separation, state)?,
                    self.stiffnesses(normal_separation, tangential_separation, state)?,
                ))
            },
        )
    }
    /// Calculates and returns the stiffnesses at a fixed state.
    fn stiffnesses(
        &self,
        normal_separation: Quantity<Length>,
        tangential_separation: Quantity<Length>,
        state: Quantity,
    ) -> Result<Stiffnesses, ConstitutiveError>;
    /// Calculates and returns the effective opening.
    fn opening(
        &self,
        normal_separation: Quantity<Length>,
        tangential_separation: Quantity<Length>,
    ) -> Result<Quantity, ConstitutiveError>;
    /// Calculates and returns the state after the separation is reached.
    fn state(
        &self,
        separation: Separation,
        normal: Normal,
        state: Quantity,
    ) -> Result<Quantity, ConstitutiveError> {
        let (normal_component, tangential_component) =
            cohesive::opening_components(&separation, &normal);
        Ok(state.max(self.opening(normal_component, tangential_component)?))
    }
    /// Returns the opening at which the interface reaches its strength, the extent of its
    /// reversible range.
    fn elastic_opening(&self) -> Quantity;
    /// Calculates and returns the error indicators of a step of the separation, from its
    /// normal and tangential components and the states at the two ends of the step.
    fn step_error(
        &self,
        separation_old: (Quantity<Length>, Quantity<Length>),
        separation: (Quantity<Length>, Quantity<Length>),
        state_old: Quantity,
        state: Quantity,
    ) -> Result<StepError, ConstitutiveError>;
}

/// Required methods for irreversible cohesive constitutive models whose tractions derive from a potential.
///
/// At a fixed state the traction is the derivative of the potential with respect to the separation,
/// so the stiffness is symmetric. The potential also includes the energy dissipated to reach the state.
pub trait HyperDamage
where
    Self: Damage,
{
    /// Calculates and returns the potential per unit area.
    fn potential(
        &self,
        normal_separation: Quantity<Length>,
        tangential_separation: Quantity<Length>,
        state: Quantity,
    ) -> Result<Quantity<ForcePerLength>, ConstitutiveError>;
}
