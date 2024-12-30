//! Cohesive constitutive models.

#[cfg(test)]
pub mod test;

mod smith_ferrante;

pub use smith_ferrante::SmithFerrante;

use super::{Constitutive, Parameters};
use crate::{
    math::Tensor,
    mechanics::{Displacement, Normal, Scalar, Stiffness, Traction, IDENTITY},
};

// TODO: loading/unloading rules using an ISV

/// Required methods for cohesive constitutive models.
pub trait Cohesive<'a>
where
    Self: Constitutive<'a>,
{
    fn calculate_traction(&self, displacement: &Displacement, normal: &Normal) -> Traction;
    fn calculate_stiffnesses(
        &self,
        displacement: &Displacement,
        normal: &Normal,
    ) -> (Stiffness, Stiffness);
}
