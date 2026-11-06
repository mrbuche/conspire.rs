//! Thermal constitutive models.

pub mod conduction;
pub mod convection;
pub mod radiation;

use core::fmt::Debug;

/// Required methods for thermal constitutive models.
pub trait Thermal
where
    Self: Clone + Debug,
{
}
