/// STEP (ISO 10303) readers.
pub mod step;

use super::brep::Brep;
use crate::io::invalid;
use std::{fs::read_to_string, io::Result, path::Path};

impl TryFrom<&Path> for Brep {
    type Error = std::io::Error;
    fn try_from(path: &Path) -> Result<Self> {
        match path.extension().and_then(|extension| extension.to_str()) {
            Some(extension)
                if extension.eq_ignore_ascii_case("step")
                    || extension.eq_ignore_ascii_case("stp") =>
            {
                step::read(&read_to_string(path)?)
            }
            _ => Err(invalid(format!("unsupported CAD file {}", path.display()))),
        }
    }
}
