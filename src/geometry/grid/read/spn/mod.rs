use crate::geometry::grid::Grid;
use std::{
    array::from_fn,
    fmt::Display,
    fs::read_to_string,
    io::{Error, ErrorKind, Result},
    path::Path,
    str::FromStr,
};

pub(super) fn read<const D: usize, T, P>(path: P, nel: Vec<usize>) -> Result<Grid<D, T>>
where
    T: FromStr,
    <T as FromStr>::Err: Display,
    P: AsRef<Path>,
{
    let nel: [usize; D] = nel.clone().try_into().map_err(|_| {
        Error::new(
            ErrorKind::InvalidInput,
            format!(
                "SPN nel has {} axes but Grid was asked for D={D}",
                nel.len()
            ),
        )
    })?;
    let text = read_to_string(path)?;
    let data = text
        .split_whitespace()
        .map(|token| {
            token
                .parse::<T>()
                .map_err(|error| Error::new(ErrorKind::InvalidData, error.to_string()))
        })
        .collect::<Result<Vec<T>>>()?;
    let total: usize = nel.iter().product();
    if data.len() != total {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "SPN file has {} values but nel {:?} expects {total}",
                data.len(),
                from_fn::<_, D, _>(|axis| nel[axis]),
            ),
        ));
    }
    Ok(Grid::new(data, nel))
}
