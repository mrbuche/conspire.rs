use crate::math::Scalar;
use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd};

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
