//! What the library has in place of the standard library, when it is built without it.
//!
//! This is for building the library as code for a device, with the nightly compiler. The
//! standard library is then not there, so the prelude is the one of `core` with the types and
//! macros of `alloc` that the library uses, and the functions of floating-point numbers that
//! `core` does not have are in [`FloatMath`](float::FloatMath).

mod float;

/// The names that every file of the library has, as `std` gives them.
///
/// The compiler does not see the use of a prelude, so it reports these as unused.
#[allow(unused_imports)]
pub(crate) mod prelude {
    pub(crate) use super::float::FloatMath;
    pub use alloc::{
        borrow::ToOwned,
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };
    pub use core::prelude::rust_2024::*;
}
