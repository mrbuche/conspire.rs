#[cfg(test)]
mod test;

use crate::io::netcdf::{
    DefineVariable, NcType, NetCDF, PutVariable,
    ffi::{
        NC_DOUBLE, NC_FLOAT, NC_GLOBAL, NC_INT, nc_close, nc_create, nc_def_dim, nc_def_var,
        nc_enddef, nc_inq_dimid, nc_inq_varid, nc_put_att_text, nc_put_var_double,
        nc_put_var_float, nc_put_var_int,
    },
};
use std::ffi::{CString, NulError, c_int, c_ulong};

// move to another folder?
impl DefineVariable for NetCDF {
    fn define_variable<T: NcType>(
        &mut self,
        name: &str,
        ndims: usize,
        dim_names: &[&str],
    ) -> Result<(), NulError> {
        let xtype = T::XTYPE;
        assert_eq!(ndims, dim_names.len(), "ndims must equal dim_names.len()");

        let name_c_str = CString::new(name)?;
        let dim_name_cstrings: Result<Vec<CString>, NulError> =
            dim_names.iter().map(|name| CString::new(*name)).collect();
        let dim_name_cstrings = dim_name_cstrings?;

        let mut dimids = Vec::with_capacity(dim_name_cstrings.len());
        for dim_name_cstr in dim_name_cstrings.iter() {
            let mut dimid: c_int = 0;
            let status = unsafe { nc_inq_dimid(self.ncid, dim_name_cstr.as_ptr(), &mut dimid) };
            assert_eq!(
                status, 0,
                "nc_inq_dimid failed for dim '{:?}' with status={status}",
                dim_name_cstr
            );
            dimids.push(dimid);
        }

        let status = unsafe {
            nc_def_var(
                self.ncid,
                name_c_str.as_ptr(),
                xtype,
                ndims as c_int,
                dimids.as_ptr(),
                &mut self.varid,
            )
        };
        assert_eq!(
            status, 0,
            "nc_def_var failed for {name} with status={status}"
        );
        Ok(())
    }
}

// move to another folder?
impl PutVariable for NetCDF {
    fn put_variable<T: NcType>(&mut self, name: &str, data: &[T]) -> Result<(), NulError> {
        let name_c_str = CString::new(name)?;
        let mut varid: c_int = 0;
        let status = unsafe { nc_inq_varid(self.ncid, name_c_str.as_ptr(), &mut varid) };
        assert_eq!(
            status, 0,
            "nc_inq_varid failed for {name} with status={status}"
        );

        let status = T::put_var(self.ncid, varid, data.as_ptr());
        assert_eq!(
            status, 0,
            "nc_put_var failed for var '{}' (varid {}) with status={status}",
            name, varid
        );
        Ok(())
    }
}

impl NcType for i32 {
    const XTYPE: c_int = NC_INT;
    fn put_var(ncid: c_int, varid: c_int, data: *const Self) -> c_int {
        unsafe { nc_put_var_int(ncid, varid, data) }
    }
}

impl NcType for f32 {
    const XTYPE: c_int = NC_FLOAT;
    fn put_var(ncid: c_int, varid: c_int, data: *const Self) -> c_int {
        unsafe { nc_put_var_float(ncid, varid, data) }
    }
}

impl NcType for f64 {
    const XTYPE: c_int = NC_DOUBLE;
    fn put_var(ncid: c_int, varid: c_int, data: *const Self) -> c_int {
        unsafe { nc_put_var_double(ncid, varid, data) }
    }
}

impl NetCDF {
    pub fn close(&mut self) {
        let status = unsafe { nc_close(self.ncid) };
        assert_eq!(status, 0, "nc_close failed with status={status}");
    }
    pub fn create(path: &str) -> Result<Self, NulError> {
        let path_c_str = CString::new(path)?;
        let mut ncid = 0;
        let status = unsafe { nc_create(path_c_str.as_ptr(), 0, &mut ncid) };
        assert_eq!(
            status, 0,
            "Might need a new error type to handle errors properly"
        );
        let dimid = 0;
        let varid = 0;
        Ok(Self { ncid, dimid, varid })
    }
    pub fn define_dimension(&mut self, name: &str, len: usize) -> Result<(), NulError> {
        let name_c_str = CString::new(name)?;
        let status = unsafe {
            nc_def_dim(
                self.ncid,
                name_c_str.as_ptr(),
                len as c_ulong,
                &mut self.dimid,
            )
        };
        assert_eq!(
            status, 0,
            "nc_def_dim failed for {name} with status={status}"
        );
        Ok(())
    }
    pub fn end_definition(&mut self) {
        let status = unsafe { nc_enddef(self.ncid) };
        assert_eq!(status, 0, "nc_enddef failed with status={status}");
    }
    pub fn global(&mut self) -> Result<(), NulError> {
        // let api_version: [f32; 1] = [8.25];
        // let file_size: [u32; 1] = [1];
        // let floating_point_word_size: [u32; 1] = [8];
        // let version: [f32; 1] = [8.25];
        self.put_attribute_text(
            "title",
            format!("{} v{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")).as_str(),
        )
    }
    pub fn put_attribute_text(&mut self, name: &str, value: &str) -> Result<(), NulError> {
        put_attribute_text(self, NC_GLOBAL, name, value)
    }
    pub fn put_variable_attribute_text(
        &mut self,
        variable: &str,
        attr_name: &str,
        value: &str,
    ) -> Result<(), NulError> {
        let variable_c_str = CString::new(variable)?;
        let mut varid: c_int = 0;
        let status = unsafe { nc_inq_varid(self.ncid, variable_c_str.as_ptr(), &mut varid) };
        assert_eq!(
            status, 0,
            "nc_inq_varid failed for {variable} with status={status}"
        );
        put_attribute_text(self, varid, attr_name, value)
    }
}

fn put_attribute_text(
    netcdf: &mut NetCDF,
    varid: c_int,
    name: &str,
    value: &str,
) -> Result<(), NulError> {
    let name_c_str = CString::new(name)?;
    let value_c_str = CString::new(value)?;
    let status = unsafe {
        nc_put_att_text(
            netcdf.ncid,
            varid,
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
