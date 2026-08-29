mod brep;

use crate::{
    geometry::cad::{brep::Brep, part_21},
    io::invalid,
};
use std::io::Result;

/// Parses STEP text and decodes every `MANIFOLD_SOLID_BREP` into a [`Brep`].
pub fn read_all(text: &str) -> Result<Vec<Brep>> {
    brep::read(&part_21::parse(text)?)
}

/// [`read_all`] for a file expected to hold exactly one solid.
pub fn read(text: &str) -> Result<Brep> {
    let mut solids = read_all(text)?;
    match solids.len() {
        1 => Ok(solids.remove(0)),
        n => Err(invalid(format!("STEP: expected one solid, found {n}"))),
    }
}
