use std::ffi::{CStr, CString, c_char, c_int, c_ulong};

// maybe an exodus module in geometry/mesh/ and another in fem/ and in vem/ etc.
// the one in mesh would be more like genesis files
// the ones in fem/ and vem/ would be more like exodus files (with output variables and timesteps)

unsafe extern "C" {
    pub fn ex_create_int(
        path: *const c_char,
        mode: c_int,
        comp_ws: *mut c_int,
        io_ws: *mut c_int,
    ) -> c_int;
    pub fn ex_close(exoid: c_int) -> c_int;
}

#[test]
fn test_exodus_create_close() {
    use std::ffi::CString;

    let path = CString::new("target/test.exo").unwrap();
    let mut comp_ws = 8;
    let mut io_ws = 8;

    let exoid = unsafe { ex_create_int(path.as_ptr(), 0, &mut comp_ws, &mut io_ws) };
    assert!(exoid >= 0, "ex_create failed with exoid={exoid}");

    let status = unsafe { ex_close(exoid) };
    assert_eq!(status, 0, "ex_close failed with status={status}");
}

unsafe extern "C" {
    pub fn nc_inq_libvers() -> *const c_char;
}

#[test]
fn test_netcdf_lib_version() {
    let ptr = unsafe { nc_inq_libvers() };
    assert!(!ptr.is_null(), "nc_inq_libvers returned null");

    let version = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .expect("netCDF version string was not valid UTF-8");

    println!("netCDF version: {version}");
    assert!(!version.is_empty(), "netCDF version string was empty");
}

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
    pub fn nc_put_var_float(ncid: c_int, varid: c_int, op: *const f32) -> c_int;
    pub fn nc_put_var_double(ncid: c_int, varid: c_int, op: *const f64) -> c_int;
}

pub const NC_FLOAT: i32 = 5;
pub const NC_DOUBLE: i32 = 6;
pub const NC_INT: i32 = 4;
pub const NC_CHAR: i32 = 2;

#[test]
fn test_netcdf() {
    let path = CString::new("target/test.nc").unwrap();
    let mut ncid = 0;

    let status = unsafe { nc_create(path.as_ptr(), 0, &mut ncid) };
    assert_eq!(status, 0);

    let name = CString::new("num_nodes").unwrap();
    let mut dimid: c_int = 0;

    let status = unsafe { nc_def_dim(ncid, name.as_ptr(), 4_u64, &mut dimid) };
    assert_eq!(status, 0);

    let dimids = [dimid];
    let mut varid: c_int = 0;
    let name = CString::new("coordx").unwrap();

    let status = unsafe {
        nc_def_var(
            ncid,
            name.as_ptr(),
            NC_DOUBLE,
            1,
            dimids.as_ptr(),
            &mut varid,
        )
    };

    let status = unsafe { nc_enddef(ncid) };
    assert_eq!(status, 0, "nc_enddef failed with status={status}");

    let values: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let status = unsafe { nc_put_var_double(ncid, varid, values.as_ptr()) };
    assert_eq!(status, 0, "nc_put_var_double failed with status={status}");

    let status = unsafe { nc_close(ncid) };
    assert_eq!(status, 0, "nc_close failed with status={status}");
}
