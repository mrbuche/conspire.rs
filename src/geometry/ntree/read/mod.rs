pub mod htg;

use crate::geometry::ntree::{Orthotree, node::split::Split};
use std::{io::Error as ErrorIO, ops::Add, path::Path};

use self::htg::ReadHtg;

pub enum Input<P>
where
    P: AsRef<Path>,
{
    Htg(P),
}

impl<P> AsRef<Path> for Input<P>
where
    P: AsRef<Path>,
{
    fn as_ref(&self) -> &Path {
        match self {
            Input::Htg(path) => path.as_ref(),
        }
    }
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, P> TryFrom<Input<P>>
    for Orthotree<D, L, M, N, T, U>
where
    P: AsRef<Path>,
    T: Add<Output = T> + Copy + Split + Into<usize> + TryFrom<usize>,
    U: Copy + From<usize> + Into<usize>,
{
    type Error = ErrorIO;
    fn try_from(input: Input<P>) -> Result<Self, Self::Error> {
        match input {
            Input::Htg(path) => Self::read_htg(path),
        }
    }
}
