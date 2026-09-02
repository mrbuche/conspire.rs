use crate::io::netcdf::NcType;

fn as_bytes<T: NcType>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * T::SIZE) }
}

pub(in crate::io::netcdf) fn encode_be<T: NcType>(data: &[T], out: &mut Vec<u8>) {
    let src = as_bytes(data);
    let start = out.len();
    out.resize(start + src.len(), 0);
    T::xdr_swap(src, &mut out[start..]);
}

#[cfg(target_endian = "little")]
pub(in crate::io::netcdf) fn encode_le<T: NcType>(data: &[T], out: &mut Vec<u8>) {
    out.extend_from_slice(as_bytes(data));
}

#[cfg(target_endian = "big")]
pub(in crate::io::netcdf) fn encode_le<T: NcType>(data: &[T], out: &mut Vec<u8>) {
    let src = as_bytes(data);
    let start = out.len();
    out.resize(start + src.len(), 0);
    T::xdr_swap(src, &mut out[start..]);
}

pub(in crate::io::netcdf) fn decode_be<T: NcType>(bytes: &[u8]) -> Vec<T> {
    let count = bytes.len() / T::SIZE;
    let mut out: Vec<T> = vec![T::default(); count];
    let dst =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), count * T::SIZE) };
    T::xdr_swap(bytes, dst);
    out
}

#[cfg(target_endian = "big")]
pub(in crate::io::netcdf) fn decode_le<T: NcType>(bytes: &[u8]) -> Vec<T> {
    decode_be::<T>(bytes)
}

#[cfg(target_endian = "little")]
pub(in crate::io::netcdf) fn decode_le<T: NcType>(bytes: &[u8]) -> Vec<T> {
    let count = bytes.len() / T::SIZE;
    let mut out: Vec<T> = Vec::with_capacity(count);
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            out.as_mut_ptr().cast::<u8>(),
            count * T::SIZE,
        );
        out.set_len(count);
    }
    out
}
