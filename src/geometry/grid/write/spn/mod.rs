use crate::geometry::grid::Grid;
use std::{
    fmt::Display,
    fs::File,
    io::{BufWriter, Result, Write},
    path::Path,
};

pub(super) fn write<const D: usize, T, P>(grid: &Grid<D, T>, path: P) -> Result<()>
where
    T: Copy + Display,
    P: AsRef<Path>,
{
    let mut file = BufWriter::new(File::create(path)?);
    grid.data_col_major()
        .iter()
        .try_for_each(|value| writeln!(file, "{value}"))
}
