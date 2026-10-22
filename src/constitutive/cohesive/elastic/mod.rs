//! Elastic cohesive constitutive models.

#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        Constitutive, ConstitutiveError,
        cohesive::{self, Cohesive, StiffnessCohesive},
    },
    math::{Quantity, TensorArray},
    mechanics::{Normal, Separation, Traction},
    units::{Length, Stress, StressPerLength},
};

/// Required methods for elastic cohesive constitutive models.
pub trait Elastic
where
    Self: Cohesive,
{
    fn traction(
        &self,
        separation: Separation,
        normal: Normal,
    ) -> Result<Traction, ConstitutiveError> {
        cohesive::traction(
            separation,
            normal,
            |normal_separation, tangential_separation| {
                self.tractions(normal_separation, tangential_separation)
            },
        )
    }
    /// Calculates and returns the normal and tangential tractions.
    fn tractions(
        &self,
        normal_separation: Quantity<Length>,
        tangential_separation: Quantity<Length>,
    ) -> Result<(Quantity<Stress>, Quantity<Stress>), ConstitutiveError>;
    fn stiffness(
        &self,
        separation: Separation,
        normal: Normal,
    ) -> Result<StiffnessCohesive, ConstitutiveError> {
        cohesive::stiffness(
            separation,
            normal,
            |normal_separation, tangential_separation| {
                let (k_nn, k_tt) = self.stiffnesses(normal_separation, tangential_separation)?;
                Ok((
                    self.tractions(normal_separation, tangential_separation)?,
                    (k_nn, Quantity::zero(), Quantity::zero(), k_tt),
                ))
            },
        )
    }
    /// Calculates and returns the normal and tangential stiffnesses.
    fn stiffnesses(
        &self,
        normal_separation: Quantity<Length>,
        tangential_separation: Quantity<Length>,
    ) -> Result<(Quantity<StressPerLength>, Quantity<StressPerLength>), ConstitutiveError>;
}

/// The linear elastic cohesive constitutive model.
#[derive(Clone, Debug)]
pub struct LinearElastic {
    /// The normal stiffness $`k_n`$.
    pub normal_stiffness: Quantity<StressPerLength>,
    /// The tangential stiffness $`k_t`$.
    pub tangential_stiffness: Quantity<StressPerLength>,
}

impl Constitutive for LinearElastic {}

impl Cohesive for LinearElastic {}

impl LinearElastic {
    /// Returns the normal stiffness.
    fn normal_stiffness(&self) -> Quantity<StressPerLength> {
        self.normal_stiffness
    }
    /// Returns the tangential stiffness.
    fn tangential_stiffness(&self) -> Quantity<StressPerLength> {
        self.tangential_stiffness
    }
}

impl Elastic for LinearElastic {
    fn tractions(
        &self,
        normal_separation: Quantity<Length>,
        tangential_separation: Quantity<Length>,
    ) -> Result<(Quantity<Stress>, Quantity<Stress>), ConstitutiveError> {
        Ok((
            self.normal_stiffness() * normal_separation,
            self.tangential_stiffness() * tangential_separation,
        ))
    }
    fn stiffnesses(
        &self,
        _normal_separation: Quantity<Length>,
        _tangential_separation: Quantity<Length>,
    ) -> Result<(Quantity<StressPerLength>, Quantity<StressPerLength>), ConstitutiveError> {
        Ok((self.normal_stiffness(), self.tangential_stiffness()))
    }
}
