use crate::geometry::grid::Grid;
use std::{
    fmt::Display,
    fs::File,
    io::{BufWriter, Result, Write},
    path::Path,
};

/// Writes a [`Grid`] to a segmentation projection (SPN) file.
///
/// Values are written one per line in column-major (`i`-fastest) order, matching
/// [`Grid`]'s internal layout.
pub(super) fn write<const D: usize, T, P>(grid: &Grid<D, T>, path: P) -> Result<()>
where
    T: Display,
    P: AsRef<Path>,
{
    let mut file = BufWriter::new(File::create(path)?);
    grid.data()
        .iter()
        .try_for_each(|value| writeln!(file, "{value}"))
}
