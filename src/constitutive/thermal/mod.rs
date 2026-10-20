//! Thermal constitutive models.

pub mod conduction;
pub mod convection;
pub mod radiation;

use std::fmt::Debug;

/// Required methods for thermal constitutive models.
pub trait Thermal
where
    Self: Clone + Debug,
{
}
