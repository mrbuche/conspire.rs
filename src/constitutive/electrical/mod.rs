//! Electrical constitutive models.

pub mod conduction;
pub mod dielectric;

use core::fmt::Debug;

/// Required methods for electrical constitutive models.
pub trait Electrical
where
    Self: Clone + Debug,
{
}
