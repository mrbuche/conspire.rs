#[cfg(target_arch = "x86_64")]
mod avx;

use crate::math::Scalar;

#[cfg(target_arch = "x86_64")]
pub(crate) fn enabled() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

/// Applies a source slice to a target slice with multiplier `w`.
pub(crate) fn axpy(target: &mut [Scalar], column: &[Scalar], w: Scalar) {
    #[cfg(target_arch = "x86_64")]
    if enabled() {
        return unsafe { avx::axpy(target, column, w) };
    }
    target
        .iter_mut()
        .zip(column.iter())
        .for_each(|(target_r, value)| *target_r -= value * w);
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
    if enabled() {
        return unsafe { avx::rank_one_quad(temp_0, temp_1, temp_2, temp_3, column, u) };
    }
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
    if enabled() {
        return unsafe { avx::rank_two_quad(temp_0, temp_1, temp_2, temp_3, column, other, u, w) };
    }
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
