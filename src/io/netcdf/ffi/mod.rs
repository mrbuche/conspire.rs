use std::ffi::{c_char, c_int};

pub const NC_GLOBAL: c_int = -1;
pub const NC_FLOAT: i32 = 5;
pub const NC_DOUBLE: i32 = 6;
pub const NC_INT: i32 = 4;
pub const NC_NOWRITE: c_int = 0;

unsafe extern "C" {
    pub fn nc_create(path: *const c_char, cmode: c_int, ncidp: *mut c_int) -> c_int;
    pub fn nc_open(path: *const c_char, mode: c_int, ncidp: *mut c_int) -> c_int;
    pub fn nc_close(ncid: c_int) -> c_int;
    pub fn nc_def_dim(ncid: c_int, name: *const c_char, len: usize, idp: *mut c_int) -> c_int;
    pub fn nc_def_var(
        ncid: c_int,
        name: *const c_char,
        xtype: c_int,
        ndims: c_int,
        dimidsp: *const c_int,
        varidp: *mut c_int,
    ) -> c_int;
    pub fn nc_enddef(ncid: c_int) -> c_int;
    pub fn nc_inq_dimid(ncid: c_int, name: *const c_char, idp: *mut c_int) -> c_int;
    pub fn nc_inq_dimlen(ncid: c_int, dimid: c_int, lenp: *mut usize) -> c_int;
    pub fn nc_inq_varid(ncid: c_int, name: *const c_char, idp: *mut c_int) -> c_int;
    pub fn nc_inq_attlen(
        ncid: c_int,
        varid: c_int,
        name: *const c_char,
        lenp: *mut usize,
    ) -> c_int;
    pub fn nc_put_att_float(
        ncid: c_int,
        varid: c_int,
        name: *const c_char,
        xtype: c_int,
        len: usize,
        op: *const f32,
    ) -> c_int;
    pub fn nc_put_att_int(
        ncid: c_int,
        varid: c_int,
        name: *const c_char,
        xtype: c_int,
        len: usize,
        op: *const c_int,
    ) -> c_int;
    pub fn nc_put_att_text(
        ncid: c_int,
        varid: c_int,
        name: *const c_char,
        len: usize,
        value: *const c_char,
    ) -> c_int;
    pub fn nc_get_att_text(
        ncid: c_int,
        varid: c_int,
        name: *const c_char,
        value: *mut c_char,
    ) -> c_int;
    pub fn nc_put_var_int(ncid: c_int, varid: c_int, op: *const c_int) -> c_int;
    pub fn nc_put_var_float(ncid: c_int, varid: c_int, op: *const f32) -> c_int;
    pub fn nc_put_var_double(ncid: c_int, varid: c_int, op: *const f64) -> c_int;
    pub fn nc_get_var_int(ncid: c_int, varid: c_int, ip: *mut c_int) -> c_int;
    pub fn nc_get_var_float(ncid: c_int, varid: c_int, fp: *mut f32) -> c_int;
    pub fn nc_get_var_double(ncid: c_int, varid: c_int, dp: *mut f64) -> c_int;
}
