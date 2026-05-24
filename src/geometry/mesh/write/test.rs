#[cfg(feature = "netcdf")]
use crate::{
    geometry::{
        Coordinates, Write,
        mesh::{TriangularMesh, from::test::mesh, write::Output},
    },
    math::test::TestError,
};

#[cfg(feature = "netcdf")]
use std::path::Path;

#[test]
#[cfg(feature = "netcdf")]
fn exodus() -> Result<(), TestError> {
    Ok(mesh().write(Output::Exodus(Path::new("target/foo.exo")))?)
}
