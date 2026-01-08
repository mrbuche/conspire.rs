//! Elastic cohesive constitutive models.

use crate::{
    constitutive::{Constitutive, ConstitutiveError, cohesive::Cohesive},
    mechanics::{Scalar, Separation, Stiffness, Traction},
};

/// Required methods for elastic cohesive constitutive models.
pub trait Elastic
where
    Self: Cohesive,
{
    fn traction(&self, separation: &Separation) -> Result<Traction, ConstitutiveError>;
    fn stiffness(&self, separation: &Separation) -> Result<Stiffness, ConstitutiveError>;
}

/// The linear elastic cohesive constitutive model.
#[derive(Clone, Debug)]
pub struct LinearElastic {
    /// The normal stiffness $`k_n`$.
    pub normal_stiffness: Scalar,
    /// The tangential stiffness $`k_t`$.
    pub tangential_stiffness: Scalar,
}

impl Constitutive for LinearElastic {}

impl Cohesive for LinearElastic {}

impl Elastic for LinearElastic {
    fn traction(&self, separation: &Separation) -> Result<Traction, ConstitutiveError> {
        Ok([
            separation[0] * self.tangential_stiffness,
            separation[1] * self.tangential_stiffness,
            separation[2] * self.normal_stiffness,
        ]
        .into())
    }
    fn stiffness(&self, _separation: &Separation) -> Result<Stiffness, ConstitutiveError> {
        Ok([
            [self.tangential_stiffness, 0.0, 0.0],
            [0.0, self.tangential_stiffness, 0.0],
            [0.0, 0.0, self.normal_stiffness],
        ]
        .into())
    }
}
