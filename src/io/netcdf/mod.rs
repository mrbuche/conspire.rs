pub mod base;
pub mod ffi;
pub mod from;

use std::ffi::c_int;

pub struct NetCDF {
    ncid: c_int,
}

impl Drop for NetCDF {
    fn drop(&mut self) {
        self.close();
    }
}
