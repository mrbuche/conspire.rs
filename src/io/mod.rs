#[cfg(feature = "netcdf")]
mod netcdf;

#[cfg(feature = "netcdf")]
pub use netcdf::{DefineVariable, GetVariable, NetCDF, PutVariable};
