#[cfg(feature = "netcdf")]
use crate::{
    geometry::mesh::{test::mesh, write::Output},
    io::Write,
    math::test::TestError,
};

#[cfg(feature = "netcdf")]
use std::path::Path;

#[test]
#[cfg(feature = "netcdf")]
fn exodus() -> Result<(), TestError> {
    Ok(mesh().write(Output::Exodus(Path::new("target/foo.exo")))?)
}
