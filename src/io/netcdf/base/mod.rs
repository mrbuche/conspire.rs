use crate::io::netcdf::{
    NetCDF,
    ffi::{
        NC_64BIT_DATA, NC_FLOAT, NC_GLOBAL, NC_INT, NC_NOWRITE, nc_close, nc_create, nc_def_dim,
        nc_enddef, nc_get_att_text, nc_inq_attlen, nc_inq_dimid, nc_inq_dimlen, nc_inq_varid,
        nc_open, nc_put_att_float, nc_put_att_int, nc_put_att_text,
    },
    nc_lock,
};
use std::ffi::{CStr, CString, NulError, c_char, c_int};

impl NetCDF {
    pub fn close(&mut self) {
        let _guard = nc_lock();
        let status = unsafe { nc_close(self.ncid) };
        assert_eq!(status, 0, "nc_close failed with status={status}");
    }
    pub fn create(path: &str) -> Result<Self, NulError> {
        let path_c_str = CString::new(path)?;
        let mut ncid = 0;
        let _guard = nc_lock();
        let status = unsafe { nc_create(path_c_str.as_ptr(), NC_64BIT_DATA, &mut ncid) };
        assert_eq!(
            status, 0,
            "Might need a new error type to handle errors properly"
        );
        let dimid = 0;
        let varid = 0;
        Ok(Self { ncid, dimid, varid })
    }
    pub fn open(path: &str) -> Result<Self, NulError> {
        let path_c_str = CString::new(path)?;
        let mut ncid = 0;
        let _guard = nc_lock();
        let status = unsafe { nc_open(path_c_str.as_ptr(), NC_NOWRITE, &mut ncid) };
        assert_eq!(status, 0, "nc_open failed for {path} with status={status}");
        Ok(Self {
            ncid,
            dimid: 0,
            varid: 0,
        })
    }
    pub fn dimension_length(&self, name: &str) -> Result<usize, NulError> {
        let name_c_str = CString::new(name)?;
        let mut dimid: c_int = 0;
        let _guard = nc_lock();
        let status = unsafe { nc_inq_dimid(self.ncid, name_c_str.as_ptr(), &mut dimid) };
        assert_eq!(
            status, 0,
            "nc_inq_dimid failed for {name} with status={status}"
        );
        let mut len: usize = 0;
        let status = unsafe { nc_inq_dimlen(self.ncid, dimid, &mut len) };
        assert_eq!(
            status, 0,
            "nc_inq_dimlen failed for {name} with status={status}"
        );
        Ok(len)
    }
    pub fn try_dimension_length(&self, name: &str) -> Result<Option<usize>, NulError> {
        let name_c_str = CString::new(name)?;
        let mut dimid: c_int = 0;
        let _guard = nc_lock();
        let status = unsafe { nc_inq_dimid(self.ncid, name_c_str.as_ptr(), &mut dimid) };
        if status != 0 {
            return Ok(None);
        }
        let mut len: usize = 0;
        let status = unsafe { nc_inq_dimlen(self.ncid, dimid, &mut len) };
        assert_eq!(
            status, 0,
            "nc_inq_dimlen failed for {name} with status={status}"
        );
        Ok(Some(len))
    }
    pub fn get_variable_attribute_text(
        &self,
        variable: &str,
        attr_name: &str,
    ) -> Result<String, NulError> {
        let variable_c_str = CString::new(variable)?;
        let mut varid: c_int = 0;
        let _guard = nc_lock();
        let status = unsafe { nc_inq_varid(self.ncid, variable_c_str.as_ptr(), &mut varid) };
        assert_eq!(
            status, 0,
            "nc_inq_varid failed for {variable} with status={status}"
        );
        let attr_c_str = CString::new(attr_name)?;
        let mut len: usize = 0;
        let status = unsafe { nc_inq_attlen(self.ncid, varid, attr_c_str.as_ptr(), &mut len) };
        assert_eq!(
            status, 0,
            "nc_inq_attlen failed for {variable}::{attr_name} with status={status}"
        );
        let mut buf: Vec<c_char> = vec![0; len];
        let status =
            unsafe { nc_get_att_text(self.ncid, varid, attr_c_str.as_ptr(), buf.as_mut_ptr()) };
        assert_eq!(
            status, 0,
            "nc_get_att_text failed for {variable}::{attr_name} with status={status}"
        );
        let bytes: Vec<u8> = buf.into_iter().map(|c| c as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
    pub fn define_dimension(&mut self, name: &str, len: usize) -> Result<(), NulError> {
        let name_c_str = CString::new(name)?;
        let _guard = nc_lock();
        let status = unsafe { nc_def_dim(self.ncid, name_c_str.as_ptr(), len, &mut self.dimid) };
        assert_eq!(
            status, 0,
            "nc_def_dim failed for {name} with status={status}"
        );
        Ok(())
    }
    pub fn end_definition(&mut self) {
        let _guard = nc_lock();
        let status = unsafe { nc_enddef(self.ncid) };
        assert_eq!(status, 0, "nc_enddef failed with status={status}");
    }
    pub fn global(&mut self) {
        self.put_global_float(c"api_version", 8.25);
        self.put_global_int(c"file_size", 1);
        self.put_global_int(c"floating_point_word_size", 8);
        self.put_global_float(c"version", 8.25);
        let title = format!("{} v{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        self.put_global_text(c"title", &title);
    }
    fn put_global_float(&mut self, name: &CStr, value: f32) {
        let _guard = nc_lock();
        let status =
            unsafe { nc_put_att_float(self.ncid, NC_GLOBAL, name.as_ptr(), NC_FLOAT, 1, &value) };
        assert_eq!(status, 0, "nc_put_att_float failed with status={status}");
    }
    fn put_global_int(&mut self, name: &CStr, value: i32) {
        let _guard = nc_lock();
        let status =
            unsafe { nc_put_att_int(self.ncid, NC_GLOBAL, name.as_ptr(), NC_INT, 1, &value) };
        assert_eq!(status, 0, "nc_put_att_int failed with status={status}");
    }
    fn put_global_text(&mut self, name: &CStr, value: &str) {
        let _guard = nc_lock();
        let status = unsafe {
            nc_put_att_text(
                self.ncid,
                NC_GLOBAL,
                name.as_ptr(),
                value.len(),
                value.as_ptr() as *const c_char,
            )
        };
        assert_eq!(status, 0, "nc_put_att_text failed with status={status}");
    }
    pub fn put_variable_attribute_text(
        &mut self,
        variable: &str,
        attr_name: &str,
        value: &str,
    ) -> Result<(), NulError> {
        let variable_c_str = CString::new(variable)?;
        let mut varid: c_int = 0;
        let _guard = nc_lock();
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
