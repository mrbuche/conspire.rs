use super::{CHUNK, Scalar};
use std::simd::{
    Simd, StdFloat,
    num::{SimdFloat, SimdUint},
};

type Half = Simd<Scalar, 4>;
const LANES: usize = CHUNK / 4;

// Unaligned quarter-row load/store, matching `avx::trisolve`'s `_mm256_loadu_pd`.
// Free `#[inline(always)]` fns rather than closures so they fold into the
// caller's `+avx2,+fma` context.
#[inline(always)]
unsafe fn ld(tile: &[Scalar], r: usize, l: usize) -> Half {
    unsafe {
        tile.as_ptr()
            .add(r * CHUNK + 4 * l)
            .cast::<Half>()
            .read_unaligned()
    }
}

#[inline(always)]
unsafe fn st(tile: &mut [Scalar], r: usize, l: usize, v: Half) {
    unsafe {
        tile.as_mut_ptr()
            .add(r * CHUNK + 4 * l)
            .cast::<Half>()
            .write_unaligned(v)
    }
}

/// `std::simd` port of [`super::avx::trisolve`], structurally identical: pivot
/// columns are fused 4 / 2 / 1 at a time so each trailing tile row is streamed
/// once per group, a group whose pivot rows are all `+0.0` is skipped, and each
/// `CHUNK`-wide row is carried as `LANES` quarter-vectors so the codegen matches
/// the intrinsic version's two independent `__m256d` chains.
// SAFETY: the x86_64 build enables avx2 + fma, so the caller must only dispatch
// here once those features are detected at runtime. Every `r` reaches `ld` / `st`
// with `r < width`, so the caller's `tile.len() == width * CHUNK` guarantee
// makes `r * CHUNK + 4 * l + 4 <= tile.len()` for `l < LANES`.
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
    unsafe {
        let mut c = 0;
        while c + 4 <= consumed {
            let uv = Half::splat(-panel[c * m + c + 1]);
            let ux = Half::splat(-panel[c * m + c + 2]);
            let uy = Half::splat(-panel[c * m + c + 3]);
            let vx = Half::splat(-panel[(c + 1) * m + c + 2]);
            let vy = Half::splat(-panel[(c + 1) * m + c + 3]);
            let xy = Half::splat(-panel[(c + 2) * m + c + 3]);
            let mut u = [Half::splat(0.0); LANES];
            let mut v = [Half::splat(0.0); LANES];
            let mut x = [Half::splat(0.0); LANES];
            let mut y = [Half::splat(0.0); LANES];
            let mut bits = Simd::<u64, 4>::splat(0);
            for l in 0..LANES {
                u[l] = ld(tile, c, l);
                v[l] = u[l].mul_add(uv, ld(tile, c + 1, l));
                st(tile, c + 1, l, v[l]);
                x[l] = v[l].mul_add(vx, u[l].mul_add(ux, ld(tile, c + 2, l)));
                st(tile, c + 2, l, x[l]);
                y[l] = x[l].mul_add(xy, v[l].mul_add(vy, u[l].mul_add(uy, ld(tile, c + 3, l))));
                st(tile, c + 3, l, y[l]);
                bits |= u[l].to_bits() | v[l].to_bits() | x[l].to_bits() | y[l].to_bits();
            }
            if bits.reduce_or() == 0 {
                c += 4;
                continue;
            }
            for r in c + 4..width {
                let first = Half::splat(-panel[c * m + r]);
                let second = Half::splat(-panel[(c + 1) * m + r]);
                let third = Half::splat(-panel[(c + 2) * m + r]);
                let fourth = Half::splat(-panel[(c + 3) * m + r]);
                for l in 0..LANES {
                    let entry = y[l].mul_add(
                        fourth,
                        x[l].mul_add(
                            third,
                            v[l].mul_add(second, u[l].mul_add(first, ld(tile, r, l))),
                        ),
                    );
                    st(tile, r, l, entry);
                }
            }
            c += 4;
        }
        while c + 2 <= consumed {
            let uv = Half::splat(-panel[c * m + c + 1]);
            let mut u = [Half::splat(0.0); LANES];
            let mut v = [Half::splat(0.0); LANES];
            let mut bits = Simd::<u64, 4>::splat(0);
            for l in 0..LANES {
                u[l] = ld(tile, c, l);
                v[l] = u[l].mul_add(uv, ld(tile, c + 1, l));
                st(tile, c + 1, l, v[l]);
                bits |= u[l].to_bits() | v[l].to_bits();
            }
            if bits.reduce_or() == 0 {
                c += 2;
                continue;
            }
            for r in c + 2..width {
                let first = Half::splat(-panel[c * m + r]);
                let second = Half::splat(-panel[(c + 1) * m + r]);
                for l in 0..LANES {
                    let entry = v[l].mul_add(second, u[l].mul_add(first, ld(tile, r, l)));
                    st(tile, r, l, entry);
                }
            }
            c += 2;
        }
        if c < consumed {
            let mut u = [Half::splat(0.0); LANES];
            let mut bits = Simd::<u64, 4>::splat(0);
            for (l, u_l) in u.iter_mut().enumerate() {
                *u_l = ld(tile, c, l);
                bits |= u_l.to_bits();
            }
            if bits.reduce_or() != 0 {
                for r in c + 1..width {
                    let first = Half::splat(-panel[c * m + r]);
                    for (l, &u_l) in u.iter().enumerate() {
                        st(tile, r, l, u_l.mul_add(first, ld(tile, r, l)));
                    }
                }
            }
        }
    }
}
