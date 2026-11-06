//! Chemical constitutive models.

pub mod diffusion;

use core::fmt::Debug;

/// Required methods for chemical constitutive models.
pub trait Chemical
where
    Self: Clone + Debug,
{
}
