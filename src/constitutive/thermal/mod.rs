//! Thermal constitutive models.

pub mod conduction;

use std::fmt::Debug;

/// Required methods for thermal constitutive models.
pub trait Thermal
where
    Self: Clone + Debug,
{
}
