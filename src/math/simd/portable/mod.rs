use crate::math::Scalar;
use std::simd::{Simd, StdFloat};

type F64x4 = Simd<Scalar, 4>;

// SAFETY (all three): the x86_64 build enables avx2 + fma, so the caller must
// only dispatch here once those features are detected at runtime; every other
// target compiles without extra features and is sound at its own baseline.

#[cfg_attr(
    target_arch = "x86_64",
    target_feature(enable = "avx2", enable = "fma")
)]
pub(super) unsafe fn axpy(target: &mut [Scalar], column: &[Scalar], w: Scalar) {
    let len = target.len();
    let spread = F64x4::splat(-w);
    let mut r = 0;
    while r + 4 <= len {
        let t = F64x4::from_slice(&target[r..]);
        F64x4::from_slice(&column[r..])
            .mul_add(spread, t)
            .copy_to_slice(&mut target[r..]);
        r += 4;
    }
    (r..len).for_each(|i| target[i] -= column[i] * w);
}

#[cfg_attr(
    target_arch = "x86_64",
    target_feature(enable = "avx2", enable = "fma")
)]
pub(super) unsafe fn rank_one_quad(
    temp_0: &mut [Scalar],
    temp_1: &mut [Scalar],
    temp_2: &mut [Scalar],
    temp_3: &mut [Scalar],
    column: &[Scalar],
    u: [Scalar; 4],
) {
    let len = column.len();
    let u_0 = F64x4::splat(u[0]);
    let u_1 = F64x4::splat(u[1]);
    let u_2 = F64x4::splat(u[2]);
    let u_3 = F64x4::splat(u[3]);
    let mut r = 0;
    while r + 4 <= len {
        let value = F64x4::from_slice(&column[r..]);
        value
            .mul_add(u_0, F64x4::from_slice(&temp_0[r..]))
            .copy_to_slice(&mut temp_0[r..]);
        value
            .mul_add(u_1, F64x4::from_slice(&temp_1[r..]))
            .copy_to_slice(&mut temp_1[r..]);
        value
            .mul_add(u_2, F64x4::from_slice(&temp_2[r..]))
            .copy_to_slice(&mut temp_2[r..]);
        value
            .mul_add(u_3, F64x4::from_slice(&temp_3[r..]))
            .copy_to_slice(&mut temp_3[r..]);
        r += 4;
    }
    (r..len).for_each(|i| {
        temp_0[i] += column[i] * u[0];
        temp_1[i] += column[i] * u[1];
        temp_2[i] += column[i] * u[2];
        temp_3[i] += column[i] * u[3];
    });
}

#[expect(clippy::too_many_arguments)]
#[cfg_attr(
    target_arch = "x86_64",
    target_feature(enable = "avx2", enable = "fma")
)]
pub(super) unsafe fn rank_two_quad(
    temp_0: &mut [Scalar],
    temp_1: &mut [Scalar],
    temp_2: &mut [Scalar],
    temp_3: &mut [Scalar],
    column: &[Scalar],
    other: &[Scalar],
    u: [Scalar; 4],
    w: [Scalar; 4],
) {
    let len = column.len();
    let u_0 = F64x4::splat(u[0]);
    let u_1 = F64x4::splat(u[1]);
    let u_2 = F64x4::splat(u[2]);
    let u_3 = F64x4::splat(u[3]);
    let w_0 = F64x4::splat(w[0]);
    let w_1 = F64x4::splat(w[1]);
    let w_2 = F64x4::splat(w[2]);
    let w_3 = F64x4::splat(w[3]);
    let mut r = 0;
    while r + 4 <= len {
        let value = F64x4::from_slice(&column[r..]);
        let second = F64x4::from_slice(&other[r..]);
        second
            .mul_add(w_0, value.mul_add(u_0, F64x4::from_slice(&temp_0[r..])))
            .copy_to_slice(&mut temp_0[r..]);
        second
            .mul_add(w_1, value.mul_add(u_1, F64x4::from_slice(&temp_1[r..])))
            .copy_to_slice(&mut temp_1[r..]);
        second
            .mul_add(w_2, value.mul_add(u_2, F64x4::from_slice(&temp_2[r..])))
            .copy_to_slice(&mut temp_2[r..]);
        second
            .mul_add(w_3, value.mul_add(u_3, F64x4::from_slice(&temp_3[r..])))
            .copy_to_slice(&mut temp_3[r..]);
        r += 4;
    }
    (r..len).for_each(|i| {
        temp_0[i] += column[i] * u[0] + other[i] * w[0];
        temp_1[i] += column[i] * u[1] + other[i] * w[1];
        temp_2[i] += column[i] * u[2] + other[i] * w[2];
        temp_3[i] += column[i] * u[3] + other[i] * w[3];
    });
}
