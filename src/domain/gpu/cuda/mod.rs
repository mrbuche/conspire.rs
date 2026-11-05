//! The little of the CUDA driver and NVRTC that explicit dynamics needs.
//!
//! Both libraries are opened when a [`Device`] is made rather than linked, so that a build needs
//! no CUDA, and a machine without it gets [`GpuError::Unavailable`] instead of a binary that
//! does not start. Kernels are compiled for the device at hand when it is opened.

use super::GpuError;
use std::{
    ffi::{CStr, CString, c_char, c_int, c_uint, c_void},
    marker::PhantomData,
    mem::{size_of, transmute_copy},
    ptr::{null, null_mut},
    sync::Arc,
};

#[cfg_attr(target_os = "linux", link(name = "dl"))]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_NOW: c_int = 2;
const COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const COMPUTE_CAPABILITY_MINOR: c_int = 76;

type Result<T> = std::result::Result<T, GpuError>;
type Context = *mut c_void;
type DevicePointer = u64;
type Program = *mut c_void;

/// An open shared library, which stays open: the function pointers taken from it must outlive
/// everything that calls them, so it is never closed.
struct Library(*mut c_void);

impl Library {
    fn open(names: &[&str]) -> Result<Self> {
        names
            .iter()
            .find_map(|name| {
                let name = CString::new(*name).ok()?;
                // SAFETY: `name` is a valid C string.
                let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
                (!handle.is_null()).then_some(Self(handle))
            })
            .ok_or_else(|| GpuError::Unavailable(format!("none of {names:?} could be opened")))
    }
    /// A function of the library, of the function pointer type `F`.
    fn function<F: Copy>(&self, name: &str) -> Result<F> {
        assert_eq!(size_of::<F>(), size_of::<*mut c_void>());
        let symbol = CString::new(name).expect("a name without nul");
        // SAFETY: the handle is open and `symbol` is a valid C string.
        let address = unsafe { dlsym(self.0, symbol.as_ptr()) };
        if address.is_null() {
            Err(GpuError::Unavailable(format!("no {name} in the library")))
        } else {
            // SAFETY: the address is that of a function, and `F` is a function pointer, which
            // the caller chooses to be of the signature that the function has.
            Ok(unsafe { transmute_copy(&address) })
        }
    }
}

/// The functions of the driver and of NVRTC.
struct Api {
    _driver: Library,
    _compiler: Library,
    init: unsafe extern "C" fn(c_uint) -> c_int,
    device_count: unsafe extern "C" fn(*mut c_int) -> c_int,
    device_name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int,
    device_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int,
    context_retain: unsafe extern "C" fn(*mut Context, c_int) -> c_int,
    context_release: unsafe extern "C" fn(c_int) -> c_int,
    context_set: unsafe extern "C" fn(Context) -> c_int,
    synchronize: unsafe extern "C" fn() -> c_int,
    alloc: unsafe extern "C" fn(*mut DevicePointer, usize) -> c_int,
    free: unsafe extern "C" fn(DevicePointer) -> c_int,
    upload: unsafe extern "C" fn(DevicePointer, *const c_void, usize) -> c_int,
    download: unsafe extern "C" fn(*mut c_void, DevicePointer, usize) -> c_int,
    fill: unsafe extern "C" fn(DevicePointer, c_uint, usize) -> c_int,
    module_load: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> c_int,
    module_unload: unsafe extern "C" fn(*mut c_void) -> c_int,
    module_function: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int,
    launch: unsafe extern "C" fn(
        *mut c_void,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> c_int,
    error_string: unsafe extern "C" fn(c_int, *mut *const c_char) -> c_int,
    program_create: unsafe extern "C" fn(
        *mut Program,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> c_int,
    program_compile: unsafe extern "C" fn(Program, c_int, *const *const c_char) -> c_int,
    program_destroy: unsafe extern "C" fn(*mut Program) -> c_int,
    binary_size: unsafe extern "C" fn(Program, *mut usize) -> c_int,
    binary: unsafe extern "C" fn(Program, *mut c_char) -> c_int,
    log_size: unsafe extern "C" fn(Program, *mut usize) -> c_int,
    log: unsafe extern "C" fn(Program, *mut c_char) -> c_int,
}

// SAFETY: the libraries are never closed and the functions are plain function pointers, which
// the driver and NVRTC document as callable from any thread.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

impl Api {
    fn load() -> Result<Self> {
        let driver = Library::open(&["libcuda.so.1", "libcuda.so"])?;
        let compiler = Library::open(&[
            "libnvrtc.so",
            "libnvrtc.so.13",
            "libnvrtc.so.12",
            "libnvrtc.so.11.2",
            "/usr/local/cuda/lib64/libnvrtc.so",
            "/usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so",
        ])?;
        Ok(Self {
            init: driver.function("cuInit")?,
            device_count: driver.function("cuDeviceGetCount")?,
            device_name: driver.function("cuDeviceGetName")?,
            device_attribute: driver.function("cuDeviceGetAttribute")?,
            context_retain: driver.function("cuDevicePrimaryCtxRetain")?,
            context_release: driver.function("cuDevicePrimaryCtxRelease_v2")?,
            context_set: driver.function("cuCtxSetCurrent")?,
            synchronize: driver.function("cuCtxSynchronize")?,
            alloc: driver.function("cuMemAlloc_v2")?,
            free: driver.function("cuMemFree_v2")?,
            upload: driver.function("cuMemcpyHtoD_v2")?,
            download: driver.function("cuMemcpyDtoH_v2")?,
            fill: driver.function("cuMemsetD32_v2")?,
            module_load: driver.function("cuModuleLoadData")?,
            module_unload: driver.function("cuModuleUnload")?,
            module_function: driver.function("cuModuleGetFunction")?,
            launch: driver.function("cuLaunchKernel")?,
            error_string: driver.function("cuGetErrorString")?,
            program_create: compiler.function("nvrtcCreateProgram")?,
            program_compile: compiler.function("nvrtcCompileProgram")?,
            program_destroy: compiler.function("nvrtcDestroyProgram")?,
            binary_size: compiler.function("nvrtcGetCUBINSize")?,
            binary: compiler.function("nvrtcGetCUBIN")?,
            log_size: compiler.function("nvrtcGetProgramLogSize")?,
            log: compiler.function("nvrtcGetProgramLog")?,
            _driver: driver,
            _compiler: compiler,
        })
    }
    /// Turns the status of a driver call into a result.
    fn check(&self, status: c_int, call: &str) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let mut message: *const c_char = null();
        // SAFETY: `message` is written with a static string, or left null.
        unsafe { (self.error_string)(status, &mut message) };
        let message = if message.is_null() {
            "unknown error".to_string()
        } else {
            // SAFETY: the driver returns a static C string.
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        };
        Err(GpuError::Cuda(format!(
            "{call} failed with {status}: {message}"
        )))
    }
}

/// A CUDA device with its context, on which buffers and kernels live.
///
/// Make it last in whatever owns buffers and kernels of it, and call [`bind`](Self::bind) from
/// whichever thread uses them.
pub struct Device {
    api: Arc<Api>,
    ordinal: c_int,
    context: Context,
    name: String,
    capability: (c_int, c_int),
}

// SAFETY: the context is a handle that is made current on whichever thread uses it.
unsafe impl Send for Device {}

impl Device {
    /// Opens the device of the given ordinal.
    pub fn new(ordinal: usize) -> Result<Self> {
        let api = Arc::new(Api::load()?);
        // SAFETY: the calls are those of the driver API, with pointers to locals.
        unsafe {
            // That the driver does not start is that this node has no device to offer, or a driver
            // that is not for it, and either way there is no device to use.
            api.check((api.init)(0), "cuInit")
                .map_err(|error| GpuError::Unavailable(error.to_string()))?;
            let mut count = 0;
            api.check((api.device_count)(&mut count), "cuDeviceGetCount")?;
            if ordinal >= count as usize {
                return Err(GpuError::Unavailable(format!(
                    "device {ordinal} was asked for and {count} are present"
                )));
            }
            let ordinal = ordinal as c_int;
            let mut name = [0 as c_char; 128];
            api.check(
                (api.device_name)(name.as_mut_ptr(), name.len() as c_int, ordinal),
                "cuDeviceGetName",
            )?;
            let name = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
            let mut major = 0;
            let mut minor = 0;
            api.check(
                (api.device_attribute)(&mut major, COMPUTE_CAPABILITY_MAJOR, ordinal),
                "cuDeviceGetAttribute",
            )?;
            api.check(
                (api.device_attribute)(&mut minor, COMPUTE_CAPABILITY_MINOR, ordinal),
                "cuDeviceGetAttribute",
            )?;
            let mut context = null_mut();
            api.check(
                (api.context_retain)(&mut context, ordinal),
                "cuDevicePrimaryCtxRetain",
            )?;
            Ok(Self {
                api,
                ordinal,
                context,
                name,
                capability: (major, minor),
            })
        }
    }
    /// The name of the device.
    #[cfg(test)]
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Makes the context of the device current on the calling thread.
    pub fn bind(&self) -> Result<()> {
        // SAFETY: the context is held by this device.
        self.api.check(
            unsafe { (self.api.context_set)(self.context) },
            "cuCtxSetCurrent",
        )
    }
    /// Waits for everything that was launched.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: nothing is passed.
        self.api
            .check(unsafe { (self.api.synchronize)() }, "cuCtxSynchronize")
    }
    /// A buffer of `len` values on the device.
    pub fn buffer<T: Copy>(&self, len: usize) -> Result<Buffer<T>> {
        let mut pointer = 0;
        // A buffer is never empty, so that it always has an address.
        let bytes = len.max(1) * size_of::<T>();
        // SAFETY: `pointer` is written by the driver.
        self.api.check(
            unsafe { (self.api.alloc)(&mut pointer, bytes) },
            "cuMemAlloc",
        )?;
        Ok(Buffer {
            api: self.api.clone(),
            pointer,
            len,
            element: PhantomData,
        })
    }
    /// A buffer on the device with the given values.
    pub fn buffer_of<T: Copy>(&self, values: &[T]) -> Result<Buffer<T>> {
        let buffer = self.buffer(values.len())?;
        buffer.upload(values)?;
        Ok(buffer)
    }
    /// Compiles `source` for this device with the given definitions, and loads it.
    pub fn module(&self, source: &str, definitions: &[String], fmad: bool) -> Result<Module> {
        let source = CString::new(source).expect("a source without nul");
        let name = CString::new("kernels.cu").expect("a name without nul");
        let mut options = vec![
            format!(
                "--gpu-architecture=sm_{}{}",
                self.capability.0, self.capability.1
            ),
            format!("--fmad={fmad}"),
            "--std=c++14".to_string(),
        ];
        // The registers that a kernel uses and the bytes that it spills to local memory, which
        // limit how many threads the device runs at once, are in the log of the compiler.
        let verbose = std::env::var_os("CONSPIRE_GPU_VERBOSE").is_some();
        if verbose {
            options.push("--ptxas-options=-v".to_string());
        }
        options.extend(
            definitions
                .iter()
                .map(|definition| format!("-D{definition}")),
        );
        let options = options
            .into_iter()
            .map(|option| CString::new(option).expect("an option without nul"))
            .collect::<Vec<_>>();
        let pointers = options
            .iter()
            .map(|option| option.as_ptr())
            .collect::<Vec<_>>();
        let api = &self.api;
        let mut program: Program = null_mut();
        // SAFETY: the strings outlive the calls, and the program is destroyed before returning.
        let compiled = unsafe {
            api.check(
                (api.program_create)(
                    &mut program,
                    source.as_ptr(),
                    name.as_ptr(),
                    0,
                    null(),
                    null(),
                ),
                "nvrtcCreateProgram",
            )?;
            let status = (api.program_compile)(program, pointers.len() as c_int, pointers.as_ptr());
            let mut log_size = 0;
            (api.log_size)(program, &mut log_size);
            let mut log = vec![0 as c_char; log_size.max(1)];
            (api.log)(program, log.as_mut_ptr());
            let log = CStr::from_ptr(log.as_ptr()).to_string_lossy().into_owned();
            if verbose && status == 0 && !log.trim().is_empty() {
                eprintln!(
                    "kernels for {} ({}):\n{log}",
                    self.name,
                    definitions.join(" ")
                );
            }
            let binary = if status == 0 {
                let mut size = 0;
                api.check((api.binary_size)(program, &mut size), "nvrtcGetCUBINSize")
                    .and_then(|()| {
                        let mut binary = vec![0 as c_char; size];
                        api.check((api.binary)(program, binary.as_mut_ptr()), "nvrtcGetCUBIN")
                            .map(|()| binary)
                    })
            } else {
                Err(GpuError::Cuda(format!(
                    "the kernels did not compile for {} with {status}:\n{log}",
                    self.name
                )))
            };
            (api.program_destroy)(&mut program);
            binary
        }?;
        let mut module = null_mut();
        // SAFETY: the binary is an image that the driver loads.
        api.check(
            unsafe { (api.module_load)(&mut module, compiled.as_ptr().cast()) },
            "cuModuleLoadData",
        )?;
        Ok(Module {
            api: self.api.clone(),
            module,
        })
    }
    /// Launches `threads` threads of a kernel in blocks of `block`, and returns without waiting.
    pub fn launch(
        &self,
        function: &Function,
        threads: usize,
        block: usize,
        arguments: &[&dyn Argument],
    ) -> Result<()> {
        if threads == 0 {
            return Ok(());
        }
        let mut pointers = arguments
            .iter()
            .map(|argument| argument.pointer())
            .collect::<Vec<_>>();
        let grid = threads.div_ceil(block) as c_uint;
        // SAFETY: the pointers point at values that live through the call, and the kernel
        // takes the arguments that it is given.
        self.api.check(
            unsafe {
                (self.api.launch)(
                    function.0,
                    grid,
                    1,
                    1,
                    block as c_uint,
                    1,
                    1,
                    0,
                    null_mut(),
                    pointers.as_mut_ptr(),
                    null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: the context was retained by this device.
        unsafe { (self.api.context_release)(self.ordinal) };
    }
}

/// A value that a kernel takes as an argument.
pub trait Argument {
    /// A pointer to the value, which the launch reads.
    fn pointer(&self) -> *mut c_void;
}

macro_rules! scalar_argument {
    ($($type:ty),*) => {$(
        impl Argument for $type {
            fn pointer(&self) -> *mut c_void {
                std::ptr::from_ref(self).cast_mut().cast()
            }
        }
    )*};
}

scalar_argument!(u32, f32, f64);

/// Values on the device.
pub struct Buffer<T> {
    api: Arc<Api>,
    pointer: DevicePointer,
    len: usize,
    element: PhantomData<T>,
}

impl<T> Argument for Buffer<T> {
    fn pointer(&self) -> *mut c_void {
        std::ptr::from_ref(&self.pointer).cast_mut().cast()
    }
}

impl<T: Copy> Buffer<T> {
    /// Copies `values` to the device, which must be as many as the buffer holds.
    pub fn upload(&self, values: &[T]) -> Result<()> {
        assert_eq!(values.len(), self.len, "a buffer is uploaded in full");
        if values.is_empty() {
            return Ok(());
        }
        // SAFETY: the buffer holds as many values as `values` does.
        self.api.check(
            unsafe {
                (self.api.upload)(
                    self.pointer,
                    values.as_ptr().cast(),
                    std::mem::size_of_val(values),
                )
            },
            "cuMemcpyHtoD",
        )
    }
    /// Copies the buffer from the device into `values`, which must be as many as it holds.
    pub fn download(&self, values: &mut [T]) -> Result<()> {
        assert_eq!(values.len(), self.len, "a buffer is downloaded in full");
        if values.is_empty() {
            return Ok(());
        }
        // SAFETY: the buffer holds as many values as `values` does.
        self.api.check(
            unsafe {
                (self.api.download)(
                    values.as_mut_ptr().cast(),
                    self.pointer,
                    std::mem::size_of_val(values),
                )
            },
            "cuMemcpyDtoH",
        )
    }
}

impl<T> Buffer<T> {
    /// Sets every 32 bits of the buffer to `value`.
    pub fn fill(&self, value: u32) -> Result<()> {
        let words = (self.len * size_of::<T>()).div_ceil(4).max(1);
        // SAFETY: the allocation is at least `words` of 32 bits, as it is at least one value.
        self.api.check(
            unsafe { (self.api.fill)(self.pointer, value, words) },
            "cuMemsetD32",
        )
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        // SAFETY: the allocation is that of this buffer.
        unsafe { (self.api.free)(self.pointer) };
    }
}

/// Compiled kernels that are loaded on the device.
pub struct Module {
    api: Arc<Api>,
    module: *mut c_void,
}

impl Module {
    /// A kernel of the module.
    pub fn function(&self, name: &str) -> Result<Function> {
        let name = CString::new(name).expect("a name without nul");
        let mut function = null_mut();
        // SAFETY: the module is loaded and `name` is a valid C string.
        self.api.check(
            unsafe { (self.api.module_function)(&mut function, self.module, name.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(Function(function))
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: the module is that loaded for this value.
        unsafe { (self.api.module_unload)(self.module) };
    }
}

/// A kernel, which lives as long as the module that it is from.
pub struct Function(*mut c_void);
