//! Domain discretization methods.

#[cfg(all(test, feature = "vem"))]
mod test;

pub(crate) mod block;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod boundary;
#[cfg(feature = "cbm")]
pub mod cbm;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
#[cfg_attr(not(feature = "fem"), allow(dead_code))]
pub(crate) mod chemical;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
#[cfg_attr(not(feature = "fem"), allow(dead_code))]
pub(crate) mod density;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
#[cfg_attr(not(feature = "fem"), allow(dead_code))]
pub(crate) mod diffusion;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
#[cfg_attr(not(feature = "fem"), allow(dead_code))]
pub(crate) mod electrical;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod factor;
#[cfg(any(feature = "elements", feature = "fem"))]
pub mod fem;
#[cfg(feature = "fem")]
#[path = "partition/feti/mod.rs"]
pub mod feti;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
pub(crate) mod from;
#[cfg(feature = "gpu")]
pub(crate) mod gpu;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
mod host;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
pub(crate) mod lumping;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
#[cfg_attr(not(feature = "fem"), allow(dead_code))]
pub(crate) mod multiphysics;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
pub(crate) mod parallel;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
pub(crate) mod solid;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
pub(crate) mod stability;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
#[cfg_attr(not(feature = "fem"), allow(dead_code))]
pub(crate) mod thermal;
pub(crate) mod time_scale;
#[cfg(feature = "vem")]
pub mod vem;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
#[cfg_attr(
    not(any(feature = "fem", feature = "vem", feature = "cbm")),
    allow(dead_code)
)]
pub(crate) mod volume;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
pub use host::*;

use crate::{
    math::{Current, Reference, TensorRank1Vec, TensorRank1Vec2D},
    units::{Acceleration, Length, Velocity},
};

/// How a scheme steps through time.
///
/// An explicit scheme has no solve, so it imposes the degrees of freedom that are held
/// itself, and an implicit one gives them to the solve of its step, which may hold them by
/// multipliers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Stepping {
    Explicit,
    Implicit,
}

pub type NodalCoordinates<const D: usize> = TensorRank1Vec<D, Current, Length>;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub type NodalCoordinatesHistory<const D: usize> = TensorRank1Vec2D<D, Current, Length>;
pub type NodalReferenceCoordinates<const D: usize> = TensorRank1Vec<D, Reference, Length>;
pub type NodalAccelerations<const D: usize> = TensorRank1Vec<D, Current, Acceleration>;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub type NodalAccelerationsHistory<const D: usize> = TensorRank1Vec2D<D, Current, Acceleration>;
pub type NodalVelocities<const D: usize> = TensorRank1Vec<D, Current, Velocity>;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub type NodalVelocitiesHistory<const D: usize> = TensorRank1Vec2D<D, Current, Velocity>;
