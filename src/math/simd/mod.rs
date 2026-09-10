#[cfg(all(not(feature = "nightly"), target_arch = "x86_64"))]
mod avx;

#[cfg(feature = "nightly")]
mod portable;

use crate::math::Scalar;

#[cfg(target_arch = "x86_64")]
use std::sync::LazyLock;

#[cfg(all(not(feature = "nightly"), target_arch = "x86_64"))]
use avx as backend;

#[cfg(all(feature = "nightly", target_arch = "x86_64"))]
use portable as backend;

/// The widest SIMD backend usable at runtime.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Isa {
    None,
    Avx2,
}

#[cfg(target_arch = "x86_64")]
static ISA: LazyLock<Isa> = LazyLock::new(|| {
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        Isa::Avx2
    } else {
        Isa::None
    }
});

/// The SIMD backend selected for this run, resolved once.
#[cfg(target_arch = "x86_64")]
pub(crate) fn isa() -> Isa {
    *ISA
}

#[cfg(any(not(feature = "nightly"), target_arch = "x86_64"))]
fn axpy_scalar(target: &mut [Scalar], column: &[Scalar], w: Scalar) {
    target
        .iter_mut()
        .zip(column.iter())
        .for_each(|(target_r, value)| *target_r -= value * w);
}

#[cfg(any(not(feature = "nightly"), target_arch = "x86_64"))]
fn rank_one_quad_scalar(
    temp_0: &mut [Scalar],
    temp_1: &mut [Scalar],
    temp_2: &mut [Scalar],
    temp_3: &mut [Scalar],
    column: &[Scalar],
    u: [Scalar; 4],
) {
    column
        .iter()
        .zip(
            temp_0
                .iter_mut()
                .zip(temp_1.iter_mut())
                .zip(temp_2.iter_mut().zip(temp_3.iter_mut())),
        )
        .for_each(|(&value, ((a_0, a_1), (a_2, a_3)))| {
            *a_0 += value * u[0];
            *a_1 += value * u[1];
            *a_2 += value * u[2];
            *a_3 += value * u[3];
        });
}

#[cfg(any(not(feature = "nightly"), target_arch = "x86_64"))]
#[expect(clippy::too_many_arguments)]
fn rank_two_quad_scalar(
    temp_0: &mut [Scalar],
    temp_1: &mut [Scalar],
    temp_2: &mut [Scalar],
    temp_3: &mut [Scalar],
    column: &[Scalar],
    other: &[Scalar],
    u: [Scalar; 4],
    w: [Scalar; 4],
) {
    column
        .iter()
        .zip(other.iter())
        .zip(
            temp_0
                .iter_mut()
                .zip(temp_1.iter_mut())
                .zip(temp_2.iter_mut().zip(temp_3.iter_mut())),
        )
        .for_each(|((&value, &second), ((a_0, a_1), (a_2, a_3)))| {
            *a_0 += value * u[0] + second * w[0];
            *a_1 += value * u[1] + second * w[1];
            *a_2 += value * u[2] + second * w[2];
            *a_3 += value * u[3] + second * w[3];
        });
}

/// Applies a source slice to a target slice with multiplier `w`.
pub(crate) fn axpy(target: &mut [Scalar], column: &[Scalar], w: Scalar) {
    #[cfg(target_arch = "x86_64")]
    if isa() == Isa::Avx2 {
        // SAFETY: `Isa::Avx2` is produced only after `is_x86_feature_detected!`
        // confirms avx2 + fma, so the backend kernel's target-feature
        // precondition holds. It reads `column.len()` elements and writes the
        // same count into `target`.
        return unsafe { backend::axpy(target, column, w) };
    }
    #[cfg(all(feature = "nightly", not(target_arch = "x86_64")))]
    // SAFETY: on a non-x86_64 target `portable::axpy` enables no features beyond
    // the platform baseline.
    return unsafe { portable::axpy(target, column, w) };
    #[cfg(not(all(feature = "nightly", not(target_arch = "x86_64"))))]
    axpy_scalar(target, column, w);
}

/// Applies a source slice to four target slices with multipliers `u`,
/// streaming the source from memory once for all four.
pub(crate) fn rank_one_quad(
    temp_0: &mut [Scalar],
    temp_1: &mut [Scalar],
    temp_2: &mut [Scalar],
    temp_3: &mut [Scalar],
    column: &[Scalar],
    u: [Scalar; 4],
) {
    #[cfg(target_arch = "x86_64")]
    if isa() == Isa::Avx2 {
        // SAFETY: `Isa::Avx2` implies avx2 + fma were detected, satisfying the
        // backend's target-feature precondition. Each `temp_*` is at least
        // `column.len()` long (caller invariant), which is all the kernel touches.
        return unsafe { backend::rank_one_quad(temp_0, temp_1, temp_2, temp_3, column, u) };
    }
    #[cfg(all(feature = "nightly", not(target_arch = "x86_64")))]
    // SAFETY: on a non-x86_64 target `portable::rank_one_quad` enables no
    // features beyond the platform baseline.
    return unsafe { portable::rank_one_quad(temp_0, temp_1, temp_2, temp_3, column, u) };
    #[cfg(not(all(feature = "nightly", not(target_arch = "x86_64"))))]
    rank_one_quad_scalar(temp_0, temp_1, temp_2, temp_3, column, u);
}

/// Applies a source slice pair to four target slices with multipliers `u`
/// and `w`, streaming both sources from memory once for all four.
#[expect(clippy::too_many_arguments)]
pub(crate) fn rank_two_quad(
    temp_0: &mut [Scalar],
    temp_1: &mut [Scalar],
    temp_2: &mut [Scalar],
    temp_3: &mut [Scalar],
    column: &[Scalar],
    other: &[Scalar],
    u: [Scalar; 4],
    w: [Scalar; 4],
) {
    #[cfg(target_arch = "x86_64")]
    if isa() == Isa::Avx2 {
        // SAFETY: `Isa::Avx2` implies avx2 + fma were detected, satisfying the
        // backend's target-feature precondition. Each `temp_*` is at least
        // `column.len()` long and `other` at least `column.len()` (caller invariant).
        return unsafe {
            backend::rank_two_quad(temp_0, temp_1, temp_2, temp_3, column, other, u, w)
        };
    }
    #[cfg(all(feature = "nightly", not(target_arch = "x86_64")))]
    // SAFETY: on a non-x86_64 target `portable::rank_two_quad` enables no
    // features beyond the platform baseline.
    return unsafe { portable::rank_two_quad(temp_0, temp_1, temp_2, temp_3, column, other, u, w) };
    #[cfg(not(all(feature = "nightly", not(target_arch = "x86_64"))))]
    rank_two_quad_scalar(temp_0, temp_1, temp_2, temp_3, column, other, u, w);
}
