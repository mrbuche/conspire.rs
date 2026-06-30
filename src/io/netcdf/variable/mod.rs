use crate::io::netcdf::{
    DefineVariable, GetVariable, NcType, NetCDF, PutVariable,
    ffi::{
        NC_DOUBLE, NC_FLOAT, NC_INT, nc_def_var, nc_get_var_double, nc_get_var_float,
        nc_get_var_int, nc_inq_dimid, nc_inq_varid, nc_put_var_double, nc_put_var_float,
        nc_put_var_int,
    },
    nc_lock,
};
use std::ffi::{CString, NulError, c_int};

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

        let _guard = nc_lock();
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

impl PutVariable for NetCDF {
    fn put_variable<T: NcType>(&mut self, name: &str, data: &[T]) -> Result<(), NulError> {
        let name_c_str = CString::new(name)?;
        let mut varid: c_int = 0;
        let _guard = nc_lock();
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

impl GetVariable for NetCDF {
    fn get_variable<T: NcType>(&self, name: &str, len: usize) -> Result<Vec<T>, NulError> {
        let name_c_str = CString::new(name)?;
        let mut varid: c_int = 0;
        let _guard = nc_lock();
        let status = unsafe { nc_inq_varid(self.ncid, name_c_str.as_ptr(), &mut varid) };
        assert_eq!(
            status, 0,
            "nc_inq_varid failed for {name} with status={status}"
        );
        let mut data: Vec<T> = vec![T::default(); len];
        let status = T::get_var(self.ncid, varid, data.as_mut_ptr());
        assert_eq!(
            status, 0,
            "nc_get_var failed for var '{name}' (varid {varid}) with status={status}"
        );
        Ok(data)
    }
    fn try_get_variable<T: NcType>(
        &self,
        name: &str,
        len: usize,
    ) -> Result<Option<Vec<T>>, NulError> {
        let name_c_str = CString::new(name)?;
        let mut varid: c_int = 0;
        let _guard = nc_lock();
        let status = unsafe { nc_inq_varid(self.ncid, name_c_str.as_ptr(), &mut varid) };
        if status != 0 {
            return Ok(None);
        }
        let mut data: Vec<T> = vec![T::default(); len];
        let status = T::get_var(self.ncid, varid, data.as_mut_ptr());
        assert_eq!(
            status, 0,
            "nc_get_var failed for var '{name}' (varid {varid}) with status={status}"
        );
        Ok(Some(data))
    }
}

impl NcType for i32 {
    const XTYPE: c_int = NC_INT;
    fn put_var(ncid: c_int, varid: c_int, data: *const Self) -> c_int {
        unsafe { nc_put_var_int(ncid, varid, data) }
    }
    fn get_var(ncid: c_int, varid: c_int, data: *mut Self) -> c_int {
        unsafe { nc_get_var_int(ncid, varid, data) }
    }
}

impl NcType for f32 {
    const XTYPE: c_int = NC_FLOAT;
    fn put_var(ncid: c_int, varid: c_int, data: *const Self) -> c_int {
        unsafe { nc_put_var_float(ncid, varid, data) }
    }
    fn get_var(ncid: c_int, varid: c_int, data: *mut Self) -> c_int {
        unsafe { nc_get_var_float(ncid, varid, data) }
    }
}

impl NcType for f64 {
    const XTYPE: c_int = NC_DOUBLE;
    fn put_var(ncid: c_int, varid: c_int, data: *const Self) -> c_int {
        unsafe { nc_put_var_double(ncid, varid, data) }
    }
    fn get_var(ncid: c_int, varid: c_int, data: *mut Self) -> c_int {
        unsafe { nc_get_var_double(ncid, varid, data) }
    }
}
