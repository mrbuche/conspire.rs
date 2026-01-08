//! Cohesive constitutive models.

pub mod elastic;

use crate::constitutive::Constitutive;

/// Required methods for cohesive constitutive models.
pub trait Cohesive
where
    Self: Constitutive,
{
}
