use crate::io::netcdf::NetCDF;
use std::{
    error::Error,
    ffi::NulError,
    fmt::{self, Debug, Display},
    path::Path,
};

pub enum NetCdfError {
    InvalidPath,
    Nul(NulError),
}

impl Debug for NetCdfError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl Display for NetCdfError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidPath => write!(f, "path is not valid UTF-8"),
            Self::Nul(error) => write!(f, "{error}"),
        }
    }
}

impl Error for NetCdfError {}

impl From<NulError> for NetCdfError {
    fn from(error: NulError) -> Self {
        Self::Nul(error)
    }
}

impl TryFrom<&Path> for NetCDF {
    type Error = NetCdfError;
    fn try_from(path: &Path) -> Result<Self, Self::Error> {
        let path = path.to_str().ok_or(NetCdfError::InvalidPath)?;
        let mut netcdf = Self::create(path)?;
        netcdf.global();
        Ok(netcdf)
    }
}
