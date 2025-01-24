use super::{Tensor, Vector};
use std::ops::Index;

// should you make types? the solvers could have default interpolators? or maybe that's too much modularity?

// ode45 allegedly uses ntrp45.m
// so maybe you need to use interpolation schemes of the same order!

/// One-dimensional linear interpolation.
pub fn interp<F, T>(x: &Vector, xp: &Vector, fp: &F) -> F
where
    F: FromIterator<T> + Index<usize, Output = T>,
    T: Tensor,
{
    let mut i = 0;
    x.iter()
        .map(|x_k| {
            i = xp.iter().position(|xp_i| xp_i > x_k).unwrap();
            (fp[i].copy() - &fp[i - 1]) / (xp[i] - xp[i - 1]) * (x_k - xp[i - 1]) + &fp[i - 1]
        })
        .collect()
}
