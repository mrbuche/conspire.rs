use std::ffi::{c_char, c_int, c_ulong};

pub const NC_GLOBAL: c_int = -1;
pub const NC_FLOAT: i32 = 5;
pub const NC_DOUBLE: i32 = 6;
pub const NC_INT: i32 = 4;
pub const NC_CHAR: i32 = 2;

unsafe extern "C" {
    pub fn nc_create(path: *const c_char, cmode: c_int, ncidp: *mut c_int) -> c_int;
    pub fn nc_close(ncid: c_int) -> c_int;
    pub fn nc_def_dim(ncid: c_int, name: *const c_char, len: c_ulong, idp: *mut c_int) -> c_int;
    pub fn nc_def_var(
        ncid: c_int,
        name: *const c_char,
        xtype: c_int,
        ndims: c_int,
        dimidsp: *const c_int,
        varidp: *mut c_int,
    ) -> c_int;
    pub fn nc_enddef(ncid: c_int) -> c_int;
    pub fn nc_put_att_float(
        ncid: c_int,
        varid: c_int,
        name: *const c_char,
        xtype: c_int,
        len: usize,
        op: *const f32,
    ) -> c_int;
    pub fn nc_put_att_text(
        ncid: c_int,
        varid: c_int,
        name: *const c_char,
        len: usize,
        value: *const c_char,
    ) -> c_int;
    pub fn nc_put_att_uint(
        ncid: c_int,
        varid: c_int,
        name: *const c_char,
        xtype: c_int,
        len: usize,
        op: *const u32,
    ) -> c_int;
    pub fn nc_put_var_int(ncid: c_int, varid: c_int, op: *const c_int) -> c_int;
    pub fn nc_put_var_float(ncid: c_int, varid: c_int, op: *const f32) -> c_int;
    pub fn nc_put_var_double(ncid: c_int, varid: c_int, op: *const f64) -> c_int;
}
