//! The functions of floating-point numbers that `core` does not have as methods.
//!
//! A device has a few of them as instructions, and the rest would be those of a math library.
//! Only what the models for a device need is here. The square root, the cube root and rounding
//! are those that `core` has, and a power that is a number of thirds, which the Neo-Hookean
//! model has, is the cube root raised to the number. The others stop the thread, so that a model
//! that needs them cannot be run on a device by mistake.

/// Stops the thread, on a device, and otherwise never returns.
fn stop() -> ! {
    #[cfg(target_arch = "nvptx64")]
    // SAFETY: stops the thread.
    unsafe {
        core::arch::nvptx::trap()
    }
    #[cfg(not(target_arch = "nvptx64"))]
    loop {}
}

/// The functions of `f64` that are in `std` and not in `core`.
pub trait FloatMath: Sized {
    fn sqrt(self) -> Self;
    fn cbrt(self) -> Self;
    fn round(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn powi(self, exponent: i32) -> Self;
    fn powf(self, exponent: Self) -> Self;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn cos(self) -> Self;
    fn acos(self) -> Self;
    fn sin(self) -> Self;
    fn sinh(self) -> Self;
    fn tanh(self) -> Self;
    fn log2(self) -> Self;
    fn hypot(self, other: Self) -> Self;
}

impl FloatMath for f64 {
    fn sqrt(self) -> f64 {
        core::f64::math::sqrt(self)
    }
    /// The cube root, to within a unit of the last place. The one of `core` is a call to a math
    /// library that a device does not have, so this is by iteration, and without a division,
    /// which on a device takes a long sequence of instructions.
    ///
    /// The reciprocal of the cube root, `y`, is found by Newton's method on `x y^3 = 1`, which
    /// is `y (4 - x y^3) / 3`, from a guess made of the exponent and part of the mantissa that
    /// is within a few percent, so that each step squares the error. The cube root is `x y^2`,
    /// and a step of Newton's method on that polishes it.
    fn cbrt(self) -> f64 {
        if self == 0.0 || !self.is_finite() {
            return self;
        }
        let magnitude = self.abs();
        let mut reciprocal =
            f64::from_bits(0x553E_F0FF_289D_D796_u64.wrapping_sub(magnitude.to_bits() / 3));
        for _ in 0..5 {
            reciprocal *= (4.0 - magnitude * reciprocal * reciprocal * reciprocal) * (1.0 / 3.0);
        }
        let squared = reciprocal * reciprocal;
        let root = magnitude * squared;
        let root = root - (root * root * root - magnitude) * squared * (1.0 / 3.0);
        if self < 0.0 { -root } else { root }
    }
    fn round(self) -> f64 {
        core::f64::math::round(self)
    }
    fn floor(self) -> f64 {
        core::f64::math::floor(self)
    }
    fn ceil(self) -> f64 {
        core::f64::math::ceil(self)
    }
    fn powi(self, exponent: i32) -> f64 {
        let mut power = 1.0;
        let mut base = self;
        let mut remaining = exponent.unsigned_abs();
        while remaining > 0 {
            if remaining & 1 == 1 {
                power *= base;
            }
            base *= base;
            remaining >>= 1;
        }
        if exponent < 0 { 1.0 / power } else { power }
    }
    /// A power that is a number of thirds, such as 5/3 and -2/3, as the cube root raised to the
    /// number. Any other power stops the thread.
    fn powf(self, exponent: f64) -> f64 {
        let thirds = exponent * 3.0;
        let nearest = if thirds >= 0.0 {
            (thirds + 0.5) as i32
        } else {
            (thirds - 0.5) as i32
        };
        let difference = thirds - nearest as f64;
        if !(-1e-9..=1e-9).contains(&difference) || nearest.abs() > 64 {
            stop()
        }
        self.cbrt().powi(nearest)
    }
    fn ln(self) -> f64 {
        stop()
    }
    fn exp(self) -> f64 {
        stop()
    }
    fn cos(self) -> f64 {
        stop()
    }
    fn acos(self) -> f64 {
        stop()
    }
    fn sin(self) -> f64 {
        stop()
    }
    fn sinh(self) -> f64 {
        stop()
    }
    fn tanh(self) -> f64 {
        stop()
    }
    fn log2(self) -> f64 {
        stop()
    }
    fn hypot(self, _other: f64) -> f64 {
        stop()
    }
}
