use crate::math::Scalar;
use std::simd::{Simd, StdFloat};

type F64x4 = Simd<Scalar, 4>;
type F64x8 = Simd<Scalar, 8>;

// SAFETY (all four kernels): the x86_64 build enables avx2 + fma, so the caller
// must only dispatch here once those features are detected at runtime; every
// other target compiles without extra features and is sound at its own baseline.
// The unaligned load/store offsets below are all guarded by a `+ LANES <= len`
// check or a caller length invariant, matching what `_mm256_loadu_pd` assumes.

#[inline(always)]
unsafe fn ld8(s: &[Scalar], r: usize) -> F64x8 {
    unsafe { s.as_ptr().add(r).cast::<F64x8>().read_unaligned() }
}

#[inline(always)]
unsafe fn ld4(s: &[Scalar], r: usize) -> F64x4 {
    unsafe { s.as_ptr().add(r).cast::<F64x4>().read_unaligned() }
}

#[inline(always)]
unsafe fn st8(s: &mut [Scalar], r: usize, v: F64x8) {
    unsafe { s.as_mut_ptr().add(r).cast::<F64x8>().write_unaligned(v) }
}

#[inline(always)]
unsafe fn st4(s: &mut [Scalar], r: usize, v: F64x4) {
    unsafe { s.as_mut_ptr().add(r).cast::<F64x4>().write_unaligned(v) }
}

#[cfg_attr(
    target_arch = "x86_64",
    target_feature(enable = "avx2", enable = "fma")
)]
pub(super) unsafe fn axpy(target: &mut [Scalar], column: &[Scalar], w: Scalar) {
    let len = target.len();
    let spread_8 = F64x8::splat(-w);
    let spread_4 = F64x4::splat(-w);
    let mut r = 0;
    unsafe {
        while r + 8 <= len {
            st8(target, r, ld8(column, r).mul_add(spread_8, ld8(target, r)));
            r += 8;
        }
        if r + 4 <= len {
            st4(target, r, ld4(column, r).mul_add(spread_4, ld4(target, r)));
            r += 4;
        }
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
    let u8 = u.map(F64x8::splat);
    let u4 = u.map(F64x4::splat);
    let mut r = 0;
    unsafe {
        while r + 8 <= len {
            let value = ld8(column, r);
            st8(temp_0, r, value.mul_add(u8[0], ld8(temp_0, r)));
            st8(temp_1, r, value.mul_add(u8[1], ld8(temp_1, r)));
            st8(temp_2, r, value.mul_add(u8[2], ld8(temp_2, r)));
            st8(temp_3, r, value.mul_add(u8[3], ld8(temp_3, r)));
            r += 8;
        }
        if r + 4 <= len {
            let value = ld4(column, r);
            st4(temp_0, r, value.mul_add(u4[0], ld4(temp_0, r)));
            st4(temp_1, r, value.mul_add(u4[1], ld4(temp_1, r)));
            st4(temp_2, r, value.mul_add(u4[2], ld4(temp_2, r)));
            st4(temp_3, r, value.mul_add(u4[3], ld4(temp_3, r)));
            r += 4;
        }
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
    let u8 = u.map(F64x8::splat);
    let w8 = w.map(F64x8::splat);
    let u4 = u.map(F64x4::splat);
    let w4 = w.map(F64x4::splat);
    let mut r = 0;
    unsafe {
        while r + 8 <= len {
            let value = ld8(column, r);
            let second = ld8(other, r);
            st8(
                temp_0,
                r,
                second.mul_add(w8[0], value.mul_add(u8[0], ld8(temp_0, r))),
            );
            st8(
                temp_1,
                r,
                second.mul_add(w8[1], value.mul_add(u8[1], ld8(temp_1, r))),
            );
            st8(
                temp_2,
                r,
                second.mul_add(w8[2], value.mul_add(u8[2], ld8(temp_2, r))),
            );
            st8(
                temp_3,
                r,
                second.mul_add(w8[3], value.mul_add(u8[3], ld8(temp_3, r))),
            );
            r += 8;
        }
        if r + 4 <= len {
            let value = ld4(column, r);
            let second = ld4(other, r);
            st4(
                temp_0,
                r,
                second.mul_add(w4[0], value.mul_add(u4[0], ld4(temp_0, r))),
            );
            st4(
                temp_1,
                r,
                second.mul_add(w4[1], value.mul_add(u4[1], ld4(temp_1, r))),
            );
            st4(
                temp_2,
                r,
                second.mul_add(w4[2], value.mul_add(u4[2], ld4(temp_2, r))),
            );
            st4(
                temp_3,
                r,
                second.mul_add(w4[3], value.mul_add(u4[3], ld4(temp_3, r))),
            );
            r += 4;
        }
    }
    (r..len).for_each(|i| {
        temp_0[i] += column[i] * u[0] + other[i] * w[0];
        temp_1[i] += column[i] * u[1] + other[i] * w[1];
        temp_2[i] += column[i] * u[2] + other[i] * w[2];
        temp_3[i] += column[i] * u[3] + other[i] * w[3];
    });
}
