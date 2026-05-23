#[cfg(feature = "netcdf")]
mod netcdf;

pub use netcdf::{DefineVariable, NetCDF, PutVariable};
