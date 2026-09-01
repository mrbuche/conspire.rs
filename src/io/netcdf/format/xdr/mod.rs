use crate::io::netcdf::NcType;

fn as_bytes<T: NcType>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * T::SIZE) }
}

pub(in crate::io::netcdf) fn encode_be<T: NcType>(data: &[T], out: &mut Vec<u8>) {
    let src = as_bytes(data);
    out.reserve(src.len());
    let start = out.len();
    unsafe { out.set_len(start + src.len()) }
    T::xdr_swap(src, &mut out[start..]);
}

pub(in crate::io::netcdf) fn decode_be<T: NcType>(bytes: &[u8]) -> Vec<T> {
    let count = bytes.len() / T::SIZE;
    let mut out: Vec<T> = Vec::with_capacity(count);
    let dst =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), count * T::SIZE) };
    T::xdr_swap(bytes, dst);
    unsafe { out.set_len(count) }
    out
}
