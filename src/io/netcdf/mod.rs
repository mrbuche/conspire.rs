#[cfg(test)]
mod test;

pub(super) mod base;
pub(super) mod format;
pub(super) mod from;
pub(super) mod variable;

pub use from::NetCdfError;

use format::{Attribute, DimSpec, Parsed, VarSpec};
use std::{
    ffi::{CString, NulError},
    fs::File,
    sync::{Mutex, MutexGuard},
};

/// Serializes netCDF operations process-wide, matching the previous behavior so
/// [`NetCDF`] can still be driven the same way from multiple threads.
static NC_LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn nc_lock() -> MutexGuard<'static, ()> {
    NC_LOCK.lock().unwrap_or_else(|error| error.into_inner())
}

/// Returns `Err` if `name` contains an interior NUL byte.
///
/// Names are no longer passed to C, but rejecting NUL keeps the observable error
/// behavior (and error type) identical for callers.
pub(crate) fn reject_nul(name: &str) -> Result<(), NulError> {
    CString::new(name).map(|_| ())
}

/// A netCDF classic-format file, open for either writing or reading.
///
/// Written files use CDF-5 ("64-bit data"); CDF-1, CDF-2 and CDF-5 are accepted
/// on read. Only fixed-size variables of `i32` / `f32` / `f64` are supported,
/// which is everything an Exodus mesh file needs.
pub struct NetCDF {
    state: State,
}

enum State {
    Write(Writer),
    Read(Reader),
}

struct Writer {
    path: String,
    dims: Vec<DimSpec>,
    global_attributes: Vec<Attribute>,
    variables: Vec<VarBuild>,
    output: Option<Output>,
}

struct VarBuild {
    name: String,
    xtype: i32,
    dim_names: Vec<String>,
    attributes: Vec<Attribute>,
}

/// State after [`NetCDF::end_definition`]: the header is on disk and every
/// variable has a resolved location.
struct Output {
    file: File,
    variables: Vec<VarSpec>,
}

struct Reader {
    bytes: Vec<u8>,
    parsed: Parsed,
}

impl Drop for NetCDF {
    fn drop(&mut self) {
        self.close();
    }
}

pub trait DefineVariable {
    fn define_variable<T: NcType>(
        &mut self,
        name: &str,
        ndims: usize,
        dim_names: &[&str],
    ) -> Result<(), NulError>;
}

pub trait PutVariable {
    fn put_variable<T: NcType>(&mut self, name: &str, data: &[T]) -> Result<(), NulError>;
}

pub trait GetVariable {
    fn get_variable<T: NcType>(&self, name: &str, len: usize) -> Result<Vec<T>, NulError>;
    fn try_get_variable<T: NcType>(
        &self,
        name: &str,
        len: usize,
    ) -> Result<Option<Vec<T>>, NulError>;
}

/// A scalar type that can be stored as netCDF external data.
///
/// Encoding and decoding are byte-oriented (see [`format::encode_be`] /
/// [`format::decode_be`]): the in-memory `[T]` is reinterpreted as bytes and
/// [`xdr_swap`](NcType::xdr_swap) does the endianness conversion as it copies
/// into (or out of) the I/O buffer — no separate pass.
///
/// # Safety
///
/// `SIZE` must equal `size_of::<Self>()`, `Self` must have no padding, and every
/// bit pattern of that width must be a valid `Self`.
pub unsafe trait NcType: Default + Copy {
    /// The netCDF external type tag (`NC_INT`, `NC_FLOAT`, `NC_DOUBLE`).
    const XTYPE: i32;
    /// Bytes per element, on disk and in memory.
    const SIZE: usize;
    /// Copy `src` to `dst` swapping each `SIZE`-byte element between native and
    /// big-endian XDR order (a byte reversal on little-endian hosts, a plain
    /// copy on big-endian ones). It is its own inverse, so both `encode_be` and
    /// `decode_be` use it. `src.len() == dst.len()`, both multiples of `SIZE`.
    fn xdr_swap(src: &[u8], dst: &mut [u8]);
}

/// Byte-reverse each `N`-byte group of `src` into `dst` (native <-> big-endian
/// XDR). `src.len() == dst.len()`, both multiples of `N`; remainders are empty.
#[inline]
fn swap_words<const N: usize>(src: &[u8], dst: &mut [u8], swap: impl Fn([u8; N]) -> [u8; N]) {
    let (src, _) = src.as_chunks::<N>();
    let (dst, _) = dst.as_chunks_mut::<N>();
    for (s, d) in src.iter().zip(dst) {
        *d = swap(*s);
    }
}

unsafe impl NcType for i32 {
    const XTYPE: i32 = format::NC_INT;
    const SIZE: usize = 4;
    #[inline]
    fn xdr_swap(src: &[u8], dst: &mut [u8]) {
        swap_words::<4>(src, dst, |w| u32::from_ne_bytes(w).to_be_bytes());
    }
}

unsafe impl NcType for f32 {
    const XTYPE: i32 = format::NC_FLOAT;
    const SIZE: usize = 4;
    #[inline]
    fn xdr_swap(src: &[u8], dst: &mut [u8]) {
        swap_words::<4>(src, dst, |w| u32::from_ne_bytes(w).to_be_bytes());
    }
}

unsafe impl NcType for f64 {
    const XTYPE: i32 = format::NC_DOUBLE;
    const SIZE: usize = 8;
    #[inline]
    fn xdr_swap(src: &[u8], dst: &mut [u8]) {
        swap_words::<8>(src, dst, |w| u64::from_ne_bytes(w).to_be_bytes());
    }
}
