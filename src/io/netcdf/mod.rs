pub mod base;
pub mod ffi;
pub mod from;
pub mod variable;

use std::ffi::{NulError, c_int};

pub struct NetCDF {
    ncid: c_int,
    dimid: c_int,
    varid: c_int,
}

impl Drop for NetCDF {
    fn drop(&mut self) {
        self.close();
    }
}

pub trait DefineVariable {
    fn define_variable<T: NcType>(
        &mut self,
        name: &str,
        ndims: usize,
        dim_names: &[&str],
    ) -> Result<(), NulError>;
}

pub trait PutVariable {
    fn put_variable<T: NcType>(&mut self, name: &str, data: &[T]) -> Result<(), NulError>;
}

pub trait GetVariable {
    fn get_variable<T: NcType>(&self, name: &str, len: usize) -> Result<Vec<T>, NulError>;
}

pub trait NcType: Default + Clone {
    const XTYPE: c_int;
    fn put_var(ncid: c_int, varid: c_int, data: *const Self) -> c_int;
    fn get_var(ncid: c_int, varid: c_int, data: *mut Self) -> c_int;
}
