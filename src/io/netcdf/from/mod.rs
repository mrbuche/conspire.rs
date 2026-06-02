use crate::io::netcdf::NetCDF;
use std::{ffi::NulError, path::Path};

impl TryFrom<&Path> for NetCDF {
    type Error = NulError;
    fn try_from(path: &Path) -> Result<Self, Self::Error> {
        let mut netcdf = Self::create(
            path.to_str()
                .expect("Might need a new error type to handle errors properly"),
        )?;
        netcdf.global()?;
        Ok(netcdf)
    }
}
