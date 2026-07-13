use super::{CHUNK, Scalar};
use std::arch::x86_64::{
    _mm256_castpd_si256, _mm256_fmadd_pd, _mm256_fnmadd_pd, _mm256_loadu_pd, _mm256_or_pd,
    _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm256_testz_si256,
};

const LANES: usize = CHUNK / 4;

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn trisolve(
    tile: &mut [Scalar],
    panel: &[Scalar],
    m: usize,
    consumed: usize,
    width: usize,
) {
    unsafe {
        let mut c = 0;
        while c + 4 <= consumed {
            let mut u = [_mm256_setzero_pd(); LANES];
            let mut v = [_mm256_setzero_pd(); LANES];
            let mut x = [_mm256_setzero_pd(); LANES];
            let mut y = [_mm256_setzero_pd(); LANES];
            let value_uv = _mm256_set1_pd(panel[c * m + c + 1]);
            let value_ux = _mm256_set1_pd(panel[c * m + c + 2]);
            let value_uy = _mm256_set1_pd(panel[c * m + c + 3]);
            let value_vx = _mm256_set1_pd(panel[(c + 1) * m + c + 2]);
            let value_vy = _mm256_set1_pd(panel[(c + 1) * m + c + 3]);
            let value_xy = _mm256_set1_pd(panel[(c + 2) * m + c + 3]);
            let mut bits = _mm256_setzero_pd();
            for l in 0..LANES {
                u[l] = _mm256_loadu_pd(tile.as_ptr().add(c * CHUNK + 4 * l));
                let row_v = tile.as_mut_ptr().add((c + 1) * CHUNK + 4 * l);
                v[l] = _mm256_fnmadd_pd(value_uv, u[l], _mm256_loadu_pd(row_v));
                _mm256_storeu_pd(row_v, v[l]);
                let row_x = tile.as_mut_ptr().add((c + 2) * CHUNK + 4 * l);
                x[l] = _mm256_fnmadd_pd(
                    value_vx,
                    v[l],
                    _mm256_fnmadd_pd(value_ux, u[l], _mm256_loadu_pd(row_x)),
                );
                _mm256_storeu_pd(row_x, x[l]);
                let row_y = tile.as_mut_ptr().add((c + 3) * CHUNK + 4 * l);
                y[l] = _mm256_fnmadd_pd(
                    value_xy,
                    x[l],
                    _mm256_fnmadd_pd(
                        value_vy,
                        v[l],
                        _mm256_fnmadd_pd(value_uy, u[l], _mm256_loadu_pd(row_y)),
                    ),
                );
                _mm256_storeu_pd(row_y, y[l]);
                bits = _mm256_or_pd(
                    bits,
                    _mm256_or_pd(_mm256_or_pd(u[l], v[l]), _mm256_or_pd(x[l], y[l])),
                );
            }
            let bits = _mm256_castpd_si256(bits);
            if _mm256_testz_si256(bits, bits) == 1 {
                c += 4;
                continue;
            }
            for r in c + 4..width {
                let first = _mm256_set1_pd(panel[c * m + r]);
                let second = _mm256_set1_pd(panel[(c + 1) * m + r]);
                let third = _mm256_set1_pd(panel[(c + 2) * m + r]);
                let fourth = _mm256_set1_pd(panel[(c + 3) * m + r]);
                for l in 0..LANES {
                    let entry = tile.as_mut_ptr().add(r * CHUNK + 4 * l);
                    _mm256_storeu_pd(
                        entry,
                        _mm256_fnmadd_pd(
                            fourth,
                            y[l],
                            _mm256_fnmadd_pd(
                                third,
                                x[l],
                                _mm256_fnmadd_pd(
                                    second,
                                    v[l],
                                    _mm256_fnmadd_pd(first, u[l], _mm256_loadu_pd(entry)),
                                ),
                            ),
                        ),
                    );
                }
            }
            c += 4;
        }
        while c + 2 <= consumed {
            let mut u = [_mm256_setzero_pd(); LANES];
            let mut v = [_mm256_setzero_pd(); LANES];
            let value = _mm256_set1_pd(panel[c * m + c + 1]);
            let mut bits = _mm256_setzero_pd();
            for l in 0..LANES {
                u[l] = _mm256_loadu_pd(tile.as_ptr().add(c * CHUNK + 4 * l));
                let next = tile.as_mut_ptr().add((c + 1) * CHUNK + 4 * l);
                v[l] = _mm256_fnmadd_pd(value, u[l], _mm256_loadu_pd(next));
                _mm256_storeu_pd(next, v[l]);
                bits = _mm256_or_pd(bits, _mm256_or_pd(u[l], v[l]));
            }
            let bits = _mm256_castpd_si256(bits);
            if _mm256_testz_si256(bits, bits) == 1 {
                c += 2;
                continue;
            }
            for r in c + 2..width {
                let first = _mm256_set1_pd(panel[c * m + r]);
                let second = _mm256_set1_pd(panel[(c + 1) * m + r]);
                for l in 0..LANES {
                    let entry = tile.as_mut_ptr().add(r * CHUNK + 4 * l);
                    _mm256_storeu_pd(
                        entry,
                        _mm256_fnmadd_pd(
                            second,
                            v[l],
                            _mm256_fnmadd_pd(first, u[l], _mm256_loadu_pd(entry)),
                        ),
                    );
                }
            }
            c += 2;
        }
        if c < consumed {
            let mut u = [_mm256_setzero_pd(); LANES];
            let mut bits = _mm256_setzero_pd();
            for (l, u_l) in u.iter_mut().enumerate() {
                *u_l = _mm256_loadu_pd(tile.as_ptr().add(c * CHUNK + 4 * l));
                bits = _mm256_or_pd(bits, *u_l);
            }
            let bits = _mm256_castpd_si256(bits);
            if _mm256_testz_si256(bits, bits) == 0 {
                for r in c + 1..width {
                    let value = _mm256_set1_pd(panel[c * m + r]);
                    for (l, &u_l) in u.iter().enumerate() {
                        let entry = tile.as_mut_ptr().add(r * CHUNK + 4 * l);
                        _mm256_storeu_pd(
                            entry,
                            _mm256_fnmadd_pd(value, u_l, _mm256_loadu_pd(entry)),
                        );
                    }
                }
            }
        }
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn rank_one_quad(
    temp_0: &mut [Scalar],
    temp_1: &mut [Scalar],
    temp_2: &mut [Scalar],
    temp_3: &mut [Scalar],
    column: &[Scalar],
    u: [Scalar; 4],
) {
    let len = column.len();
    let u_0 = _mm256_set1_pd(u[0]);
    let u_1 = _mm256_set1_pd(u[1]);
    let u_2 = _mm256_set1_pd(u[2]);
    let u_3 = _mm256_set1_pd(u[3]);
    let mut r = 0;
    unsafe {
        while r + 4 <= len {
            let value = _mm256_loadu_pd(column.as_ptr().add(r));
            _mm256_storeu_pd(
                temp_0.as_mut_ptr().add(r),
                _mm256_fmadd_pd(value, u_0, _mm256_loadu_pd(temp_0.as_ptr().add(r))),
            );
            _mm256_storeu_pd(
                temp_1.as_mut_ptr().add(r),
                _mm256_fmadd_pd(value, u_1, _mm256_loadu_pd(temp_1.as_ptr().add(r))),
            );
            _mm256_storeu_pd(
                temp_2.as_mut_ptr().add(r),
                _mm256_fmadd_pd(value, u_2, _mm256_loadu_pd(temp_2.as_ptr().add(r))),
            );
            _mm256_storeu_pd(
                temp_3.as_mut_ptr().add(r),
                _mm256_fmadd_pd(value, u_3, _mm256_loadu_pd(temp_3.as_ptr().add(r))),
            );
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

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn rank_two_quad(
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
    let u_0 = _mm256_set1_pd(u[0]);
    let u_1 = _mm256_set1_pd(u[1]);
    let u_2 = _mm256_set1_pd(u[2]);
    let u_3 = _mm256_set1_pd(u[3]);
    let w_0 = _mm256_set1_pd(w[0]);
    let w_1 = _mm256_set1_pd(w[1]);
    let w_2 = _mm256_set1_pd(w[2]);
    let w_3 = _mm256_set1_pd(w[3]);
    let mut r = 0;
    unsafe {
        while r + 4 <= len {
            let value = _mm256_loadu_pd(column.as_ptr().add(r));
            let second = _mm256_loadu_pd(other.as_ptr().add(r));
            _mm256_storeu_pd(
                temp_0.as_mut_ptr().add(r),
                _mm256_fmadd_pd(
                    second,
                    w_0,
                    _mm256_fmadd_pd(value, u_0, _mm256_loadu_pd(temp_0.as_ptr().add(r))),
                ),
            );
            _mm256_storeu_pd(
                temp_1.as_mut_ptr().add(r),
                _mm256_fmadd_pd(
                    second,
                    w_1,
                    _mm256_fmadd_pd(value, u_1, _mm256_loadu_pd(temp_1.as_ptr().add(r))),
                ),
            );
            _mm256_storeu_pd(
                temp_2.as_mut_ptr().add(r),
                _mm256_fmadd_pd(
                    second,
                    w_2,
                    _mm256_fmadd_pd(value, u_2, _mm256_loadu_pd(temp_2.as_ptr().add(r))),
                ),
            );
            _mm256_storeu_pd(
                temp_3.as_mut_ptr().add(r),
                _mm256_fmadd_pd(
                    second,
                    w_3,
                    _mm256_fmadd_pd(value, u_3, _mm256_loadu_pd(temp_3.as_ptr().add(r))),
                ),
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
