//! Chemical constitutive models.

pub mod diffusion;

use std::fmt::Debug;

/// Required methods for chemical constitutive models.
pub trait Chemical
where
    Self: Clone + Debug,
{
}
