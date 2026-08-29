/// STEP (ISO 10303) readers.
pub mod step;

use super::brep::Brep;
use crate::io::invalid;
use std::{fs::read_to_string, io::Result, path::Path};

fn is_step(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("step") || extension.eq_ignore_ascii_case("stp")
        })
}

impl TryFrom<&Path> for Brep {
    type Error = std::io::Error;
    fn try_from(path: &Path) -> Result<Self> {
        if is_step(path) {
            step::read(&read_to_string(path)?)
        } else {
            Err(invalid(format!("unsupported CAD file {}", path.display())))
        }
    }
}

/// Every solid in a CAD file, in file order.
pub fn read_solids(path: &Path) -> Result<Vec<Brep>> {
    if is_step(path) {
        step::read_all(&read_to_string(path)?)
    } else {
        Err(invalid(format!("unsupported CAD file {}", path.display())))
    }
}
