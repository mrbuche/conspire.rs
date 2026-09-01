//! Variable-data conversion between native memory order and netCDF's big-endian
//! XDR encoding, fused with the buffer copy.

use crate::io::netcdf::NcType;

/// Reinterpret `[T]` as bytes. Sound for every [`NcType`] by that trait's
/// safety contract (`SIZE == size_of::<T>()`, no padding, all bit patterns valid).
fn as_bytes<T: NcType>(data: &[T]) -> &[u8] {
    // SAFETY: see the [`NcType`] contract; `u8` alignment (1) is always satisfied.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * T::SIZE) }
}

/// Append `data` to `out` as big-endian XDR bytes.
///
/// The endianness swap happens *inside* the single copy from `data` into `out`
/// (no separate pass), and the destination bytes are made live with `set_len`
/// and fully written by [`NcType::xdr_swap`] before being read (no zero-fill).
pub(in crate::io::netcdf) fn encode_be<T: NcType>(data: &[T], out: &mut Vec<u8>) {
    let src = as_bytes(data);
    out.reserve(src.len());
    let start = out.len();
    // SAFETY: `reserve` guaranteed the capacity and `xdr_swap` writes every byte
    // of `start..start + src.len()` before anything reads it.
    unsafe { out.set_len(start + src.len()) }
    T::xdr_swap(src, &mut out[start..]);
}

/// Decode `bytes` (big-endian XDR, length a multiple of `T::SIZE`) into `Vec<T>`.
pub(in crate::io::netcdf) fn decode_be<T: NcType>(bytes: &[u8]) -> Vec<T> {
    let count = bytes.len() / T::SIZE;
    let mut out: Vec<T> = Vec::with_capacity(count);
    // SAFETY: capacity for `count` elements just reserved; `xdr_swap` writes every
    // byte before `set_len` exposes them, and `u8` alignment is trivially met.
    let dst =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), count * T::SIZE) };
    T::xdr_swap(bytes, dst);
    // SAFETY: all `count` elements were initialized above.
    unsafe { out.set_len(count) }
    out
}
