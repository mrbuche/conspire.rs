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
                let data = if npy.fortran_order {
                    npy.data
                } else {
                    transpose(npy.data, nel)
                };
                Ok(Grid::new(data, nel))
            }
            Input::Vti(path) => vti::read(path),
        }
    }
}

fn transpose<const D: usize, T: Copy>(c_order: Vec<T>, nel: [usize; D]) -> Vec<T> {
    let mut axis_0_stride = [1usize; D];
    for axis in 1..D {
        axis_0_stride[axis] = axis_0_stride[axis - 1] * nel[axis - 1];
    }
    let mut c_stride = [1usize; D];
    for axis in (0..D.saturating_sub(1)).rev() {
        c_stride[axis] = c_stride[axis + 1] * nel[axis + 1];
    }
    (0..nel.iter().product())
        .map(|flat| {
            let c_flat = (0..D)
                .map(|axis| (flat / axis_0_stride[axis]) % nel[axis] * c_stride[axis])
                .sum::<usize>();
            c_order[c_flat]
        })
        .collect()
}
