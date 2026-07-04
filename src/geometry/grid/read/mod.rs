mod spn;
mod vti;

use crate::{
    geometry::grid::Grid,
    io::{Npy, NpyType},
};
use std::{
    fmt::Display,
    io::{Error as ErrorIO, ErrorKind},
    path::Path,
    str::FromStr,
};

pub enum Input<P>
where
    P: AsRef<Path>,
{
    Npy(P),
    Spn(P, Vec<usize>),
    Vti(P),
}

impl<P> AsRef<Path> for Input<P>
where
    P: AsRef<Path>,
{
    fn as_ref(&self) -> &Path {
        match self {
            Input::Npy(path) => path.as_ref(),
            Input::Spn(path, _) => path.as_ref(),
            Input::Vti(path) => path.as_ref(),
        }
    }
}

impl<const D: usize, T, P> TryFrom<Input<P>> for Grid<D, T>
where
    P: AsRef<Path>,
    T: NpyType + FromStr,
    <T as FromStr>::Err: Display,
{
    type Error = ErrorIO;
    fn try_from(input: Input<P>) -> Result<Self, Self::Error> {
        match input {
            Input::Spn(path, nel) => spn::read(path, nel),
            Input::Npy(path) => {
                let npy = Npy::<T>::read(path)?;
                let nel: [usize; D] = npy.shape.try_into().map_err(|shape: Vec<usize>| {
                    ErrorIO::new(
                        ErrorKind::InvalidData,
                        format!("npy has {} axes but Grid was asked for D={D}", shape.len()),
                    )
                })?;
                if npy.fortran_order {
                    Ok(Grid::new(npy.data, nel))
                } else {
                    Ok(Grid::new_row_major(npy.data, nel))
                }
            }
            Input::Vti(path) => vti::read(path),
        }
    }
}

pub(super) fn transpose<const D: usize, T: Copy>(c_order: Vec<T>, nel: [usize; D]) -> Vec<T> {
    let total: usize = nel.iter().product();
    let mut f_stride = [1usize; D];
    for axis in 1..D {
        f_stride[axis] = f_stride[axis - 1] * nel[axis - 1];
    }
    let mut c_stride = [1usize; D];
    for axis in (0..D.saturating_sub(1)).rev() {
        c_stride[axis] = c_stride[axis + 1] * nel[axis + 1];
    }
    let mut out: Vec<T> = Vec::with_capacity(total);
    if total > 0 {
        unsafe {
            transpose_block(&c_order, out.as_mut_ptr(), 0, 0, nel, &c_stride, &f_stride);
            out.set_len(total);
        }
    }
    out
}

unsafe fn transpose_block<const D: usize, T: Copy>(
    src: &[T],
    dst: *mut T,
    c_off: usize,
    f_off: usize,
    len: [usize; D],
    c_stride: &[usize; D],
    f_stride: &[usize; D],
) {
    const TILE: usize = 16;
    if let Some(axis) = (0..D)
        .filter(|&axis| len[axis] > TILE)
        .max_by_key(|&axis| len[axis])
    {
        let half = len[axis] / 2;
        let mut lo = len;
        lo[axis] = half;
        let mut hi = len;
        hi[axis] = len[axis] - half;
        unsafe {
            transpose_block(src, dst, c_off, f_off, lo, c_stride, f_stride);
            transpose_block(
                src,
                dst,
                c_off + half * c_stride[axis],
                f_off + half * f_stride[axis],
                hi,
                c_stride,
                f_stride,
            );
        }
        return;
    }
    let volume: usize = len.iter().product();
    let mut idx = [0usize; D];
    let (mut c, mut f) = (c_off, f_off);
    for _ in 0..volume {
        unsafe {
            *dst.add(f) = src[c];
        }
        for axis in 0..D {
            idx[axis] += 1;
            c += c_stride[axis];
            f += f_stride[axis];
            if idx[axis] < len[axis] {
                break;
            }
            idx[axis] = 0;
            c -= len[axis] * c_stride[axis];
            f -= len[axis] * f_stride[axis];
        }
    }
}
