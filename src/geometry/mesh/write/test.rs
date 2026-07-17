#[cfg(feature = "netcdf")]
use crate::{
    geometry::mesh::{test::mesh, write::Output},
    io::Write,
    math::assert::AssertionError,
};

#[cfg(feature = "netcdf")]
use std::path::Path;

#[test]
#[cfg(feature = "netcdf")]
fn exodus() -> Result<(), AssertionError> {
    Ok(mesh().write(Output::Exodus(Path::new("target/foo.exo")))?)
}
