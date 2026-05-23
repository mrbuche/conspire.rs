use std::{env::var_os, path::Path};

const LIB_DIRS: [&str; 2] = ["/opt/seacas/lib", "/usr/lib/x86_64-linux-gnu"];

fn main() {
    if var_os("CARGO_FEATURE_NETCDF").is_some() {
        for lib_dir in LIB_DIRS {
            let netcdf = format!("{}/libnetcdf.so", lib_dir);
            if Path::new(&netcdf).exists() {
                println!("cargo:rustc-link-search=native={}", lib_dir);
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir);
                println!("cargo:rustc-link-lib=exodus");
                println!("cargo:rustc-link-lib=netcdf");
                return;
            }
        }
        panic!("Could not find netCDF library in known locations");
    }
}
