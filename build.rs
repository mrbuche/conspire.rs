use std::{env, path::Path};

fn main() {
    if env::var_os("CARGO_FEATURE_NETCDF").is_none() {
        return;
    }
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let (lib_dirs, lib_file): (&[&str], &str) = match target_os.as_str() {
        "linux" => (&["/usr/lib/x86_64-linux-gnu"], "libnetcdf.so"),
        "macos" => (&["/opt/homebrew/lib", "/usr/local/lib"], "libnetcdf.dylib"),
        "windows" => (&["C:/vcpkg/installed/x64-windows/lib"], "netcdf.lib"),
        other => panic!("netCDF feature is not configured for target OS '{other}'"),
    };
    for lib_dir in lib_dirs {
        if Path::new(&format!("{lib_dir}/{lib_file}")).exists() {
            println!("cargo:rustc-link-search=native={lib_dir}");
            if target_os != "windows" {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
            }
            println!("cargo:rustc-link-lib=netcdf");
            return;
        }
    }
    panic!("Could not find netCDF library in known locations");
}
