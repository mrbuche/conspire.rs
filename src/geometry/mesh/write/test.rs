use crate::{
    geometry::mesh::{
        test::mesh,
        write::{Output, exodus::ExodusFormat},
    },
    io::Write,
    math::assert::AssertionError,
};

use std::path::Path;

#[test]
fn exodus() -> Result<(), AssertionError> {
    Ok(mesh().write(Output::Exodus(ExodusFormat::Classic(Path::new(
        "target/foo.exo",
    ))))?)
}
