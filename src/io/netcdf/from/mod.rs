use crate::io::netcdf::NetCDF;
use std::{ffi::NulError, path::Path};

impl TryFrom<&Path> for NetCDF {
    type Error = NulError;
    fn try_from(path: &Path) -> Result<Self, Self::Error> {
        let path = path
            .to_str()
            .expect("Might need a new error type to handle errors properly");
        let mut netcdf = Self::create(path)?;
        netcdf.global();
        Ok(netcdf)
    }
}
