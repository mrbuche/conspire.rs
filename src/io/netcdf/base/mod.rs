#[cfg(test)]
mod test;

use crate::io::netcdf::{
    NetCDF,
    ffi::{
        NC_FLOAT, NC_GLOBAL, NC_INT, nc_close, nc_create, nc_def_dim, nc_def_var, nc_enddef,
        nc_put_att_float, nc_put_att_text, nc_put_att_uint, nc_put_var_double, nc_put_var_int,
    },
};
use std::ffi::{CString, NulError, c_char, c_int, c_ulong};

impl NetCDF {
    pub fn add_attribute_text(&mut self, name: &str, value: &str) -> Result<(), NulError> {
        let name_c_str = CString::new(name)?;
        let value_c_str = CString::new(value)?;
        let status = unsafe {
            nc_put_att_text(
                self.ncid,
                NC_GLOBAL,
                name_c_str.as_ptr(),
                value_c_str.as_bytes().len(),
                value_c_str.as_ptr(),
            )
        };
        assert_eq!(
            status, 0,
            "nc_put_att_text failed for {name} with status={status}"
        );
        Ok(())
    }
    pub fn close(&mut self) {
        let status = unsafe { nc_close(self.ncid) };
        assert_eq!(status, 0, "nc_close failed with status={status}");
    }
    pub fn create(path: &str) -> Result<Self, NulError> {
        let path = CString::new(path)?;
        let mut ncid = 0;
        let status = unsafe { nc_create(path.as_ptr(), 0, &mut ncid) };
        assert_eq!(
            status, 0,
            "Might need a new error type to handle errors properly"
        );
        Ok(Self { ncid })
    }
    pub fn global(&mut self) -> Result<(), NulError> {
        // let api_version: [f32; 1] = [8.25];
        // let file_size: [u32; 1] = [1];
        // let floating_point_word_size: [u32; 1] = [8];
        // let version: [f32; 1] = [8.25];
        self.add_attribute_text(
            "title",
            format!("{} v{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")).as_str(),
        )
    }
}
