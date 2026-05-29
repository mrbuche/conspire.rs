#[cfg(feature = "netcdf")]
pub mod exodus;

#[cfg(feature = "netcdf")]
pub use self::exodus::ReadExodus;