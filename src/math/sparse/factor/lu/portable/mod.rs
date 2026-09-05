use super::{CHUNK, Scalar};
use std::simd::{
    Simd, StdFloat,
    num::{SimdFloat, SimdUint},
};

type F64s = Simd<Scalar, CHUNK>;

/// `std::simd` port of [`super::avx::trisolve`], structurally identical: pivot
/// columns are fused 4 / 2 / 1 at a time so each trailing tile row is streamed
/// once per group, and a group whose pivot rows are all `+0.0` is skipped.
/// One `F64s` spans a whole `CHUNK`-wide row, so there is no per-lane loop.
// SAFETY: the x86_64 build enables avx2 + fma, so the caller must only dispatch
// here once those features are detected at runtime.
#[cfg_attr(
    target_arch = "x86_64",
    target_feature(enable = "avx2", enable = "fma")
)]
pub(super) unsafe fn trisolve(
    tile: &mut [Scalar],
    panel: &[Scalar],
    m: usize,
    consumed: usize,
    width: usize,
) {
    let load = |tile: &[Scalar], r: usize| F64s::from_slice(&tile[r * CHUNK..]);
    let store = |tile: &mut [Scalar], r: usize, val: F64s| {
        val.copy_to_slice(&mut tile[r * CHUNK..r * CHUNK + CHUNK])
    };
    let mut c = 0;
    while c + 4 <= consumed {
        let uv = Simd::splat(-panel[c * m + c + 1]);
        let ux = Simd::splat(-panel[c * m + c + 2]);
        let uy = Simd::splat(-panel[c * m + c + 3]);
        let vx = Simd::splat(-panel[(c + 1) * m + c + 2]);
        let vy = Simd::splat(-panel[(c + 1) * m + c + 3]);
        let xy = Simd::splat(-panel[(c + 2) * m + c + 3]);
        let u = load(tile, c);
        let v = u.mul_add(uv, load(tile, c + 1));
        store(tile, c + 1, v);
        let x = v.mul_add(vx, u.mul_add(ux, load(tile, c + 2)));
        store(tile, c + 2, x);
        let y = x.mul_add(xy, v.mul_add(vy, u.mul_add(uy, load(tile, c + 3))));
        store(tile, c + 3, y);
        if (u.to_bits() | v.to_bits() | x.to_bits() | y.to_bits()).reduce_or() == 0 {
            c += 4;
            continue;
        }
        for r in c + 4..width {
            let first = Simd::splat(-panel[c * m + r]);
            let second = Simd::splat(-panel[(c + 1) * m + r]);
            let third = Simd::splat(-panel[(c + 2) * m + r]);
            let fourth = Simd::splat(-panel[(c + 3) * m + r]);
            let entry = y.mul_add(
                fourth,
                x.mul_add(third, v.mul_add(second, u.mul_add(first, load(tile, r)))),
            );
            store(tile, r, entry);
        }
        c += 4;
    }
    while c + 2 <= consumed {
        let uv = Simd::splat(-panel[c * m + c + 1]);
        let u = load(tile, c);
        let v = u.mul_add(uv, load(tile, c + 1));
        store(tile, c + 1, v);
        if (u.to_bits() | v.to_bits()).reduce_or() == 0 {
            c += 2;
            continue;
        }
        for r in c + 2..width {
            let first = Simd::splat(-panel[c * m + r]);
            let second = Simd::splat(-panel[(c + 1) * m + r]);
            let entry = v.mul_add(second, u.mul_add(first, load(tile, r)));
            store(tile, r, entry);
        }
        c += 2;
    }
    if c < consumed {
        let u = load(tile, c);
        if u.to_bits().reduce_or() != 0 {
            for r in c + 1..width {
                let first = Simd::splat(-panel[c * m + r]);
                store(tile, r, u.mul_add(first, load(tile, r)));
            }
        }
    }
}
