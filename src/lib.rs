#![doc = include_str!("../README.md")]
#![cfg_attr(feature = "autodiff", feature(autodiff))]
#![cfg_attr(feature = "nightly", feature(portable_simd))]
// Paths come from `core` or `alloc` when they have the item, and from `std` only for what they
// do not, so that the code that does not need an operating system does not depend on one.
#![warn(
    clippy::alloc_instead_of_core,
    clippy::std_instead_of_alloc,
    clippy::std_instead_of_core
)]

extern crate alloc;

#[cfg(feature = "cbm")]
#[doc(hidden)]
pub use domain::cbm;

#[cfg(feature = "constitutive")]
pub mod constitutive;

#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
#[path = "domain/mod.rs"]
pub mod domain;

#[cfg(feature = "fem")]
#[doc(hidden)]
pub use domain::fem;

#[cfg(feature = "fem")]
#[doc(hidden)]
pub use domain::feti;

#[cfg(feature = "geometry")]
pub mod geometry;

#[cfg(feature = "io")]
pub mod io;

#[cfg(feature = "math")]
pub mod math;

#[cfg(feature = "mechanics")]
pub mod mechanics;

#[cfg(feature = "physics")]
pub mod physics;

#[cfg(feature = "math")]
pub mod units;

#[cfg(feature = "vem")]
#[doc(hidden)]
pub use domain::vem;

/// Absolute tolerance.
pub const ABS_TOL: f64 = 1e-12;

/// Relative tolerance.
pub const REL_TOL: f64 = 1e-12;

/// A perturbation.
pub const EPSILON: f64 = 1e-6;
