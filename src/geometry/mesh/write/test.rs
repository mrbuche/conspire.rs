use crate::{
    geometry::mesh::{test::mesh, write::Output},
    io::Write,
    math::assert::AssertionError,
};

use std::path::Path;

#[test]
fn exodus() -> Result<(), AssertionError> {
    Ok(mesh().write(Output::Exodus(Path::new("target/foo.exo")))?)
}
