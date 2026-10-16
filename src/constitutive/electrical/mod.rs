//! Electrical constitutive models.

pub mod dielectric;

use std::fmt::Debug;

/// Required methods for electrical constitutive models.
pub trait Electrical
where
    Self: Clone + Debug,
{
}
