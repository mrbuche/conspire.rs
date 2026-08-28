mod brep;
#[cfg(test)]
mod test;

use crate::geometry::cad::{brep::Brep, part_21};
use std::io::Result;

/// Parses STEP text and decodes its `MANIFOLD_SOLID_BREP` into a [`Brep`].
pub fn read(text: &str) -> Result<Brep> {
    brep::read(&part_21::parse(text)?)
}
