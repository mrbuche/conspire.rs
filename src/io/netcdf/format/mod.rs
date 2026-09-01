//! Minimal reader and writer for the netCDF *classic* formats (CDF-1, CDF-2,
//! CDF-5), sufficient for Exodus mesh files: fixed-size (non-record) variables
//! only, whole-variable access, `int` / `float` / `double` data.
//!
//! Files are always written as CDF-5 ("64-bit data"); all three classic
//! variants are accepted on read. NetCDF-4 (HDF5) files are not supported.

#[cfg(test)]
mod test;

pub(super) const NC_INT: i32 = 4;
pub(super) const NC_FLOAT: i32 = 5;
pub(super) const NC_DOUBLE: i32 = 6;

const NC_DIMENSION: i32 = 0x0A;
const NC_VARIABLE: i32 = 0x0B;
const NC_ATTRIBUTE: i32 = 0x0C;

/// Bytes on disk for one element of the given external type.
pub(super) fn type_size(xtype: i32) -> usize {
    match xtype {
        NC_INT | NC_FLOAT => 4,
        NC_DOUBLE => 8,
        _ => panic!("unsupported netCDF external type {xtype}"),
    }
}

/// Round `n` up to the next multiple of four (classic-format alignment).
fn pad4(n: u64) -> u64 {
    n.div_ceil(4) * 4
}

pub(super) enum AttValue {
    Int(Vec<i32>),
    Float(Vec<f32>),
    Text(String),
}

pub(super) struct Attribute {
    pub name: String,
    pub value: AttValue,
}

pub(super) struct DimSpec {
    pub name: String,
    pub len: u64,
}

pub(super) struct VarSpec {
    pub name: String,
    pub xtype: i32,
    pub dimids: Vec<usize>,
    pub atts: Vec<Attribute>,
    /// Byte offset of this variable's data from the start of the file.
    pub begin: u64,
    /// Bytes occupied on disk (data length padded to a multiple of four).
    pub vsize: u64,
}

impl VarSpec {
    /// Number of scalar elements, i.e. the product of the referenced dimension
    /// lengths (an empty dimension list denotes a scalar: one element).
    pub fn elements(&self, dims: &[DimSpec]) -> u64 {
        self.dimids
            .iter()
            .map(|&d| dims[d].len)
            .product::<u64>()
            .max(1)
    }
}

// ------------------------------------------------------------------------------
// Writing (CDF-5 only)
// ------------------------------------------------------------------------------

struct HeaderWriter {
    buf: Vec<u8>,
}

impl HeaderWriter {
    fn tag(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }
    fn raw(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }
    fn align(&mut self) {
        while !self.buf.len().is_multiple_of(4) {
            self.buf.push(0);
        }
    }
    fn name(&mut self, s: &str) {
        self.i64(s.len() as i64); // CDF-5: name length is 8 bytes
        self.raw(s.as_bytes());
        self.align();
    }
    fn att_list(&mut self, atts: &[Attribute]) {
        if atts.is_empty() {
            self.tag(0); // ABSENT
            self.i64(0);
            return;
        }
        self.tag(NC_ATTRIBUTE);
        self.i64(atts.len() as i64);
        for att in atts {
            self.name(&att.name);
            match &att.value {
                AttValue::Int(xs) => {
                    self.tag(NC_INT);
                    self.i64(xs.len() as i64);
                    for &x in xs {
                        self.raw(&x.to_be_bytes());
                    }
                }
                AttValue::Float(xs) => {
                    self.tag(NC_FLOAT);
                    self.i64(xs.len() as i64);
                    for &x in xs {
                        self.raw(&x.to_bits().to_be_bytes());
                    }
                }
                AttValue::Text(s) => {
                    self.tag(NC_CHAR_TYPE);
                    self.i64(s.len() as i64);
                    self.raw(s.as_bytes());
                }
            }
            self.align();
        }
    }
}

const NC_CHAR_TYPE: i32 = 2;

fn encode_header(dims: &[DimSpec], gatts: &[Attribute], vars: &[VarSpec]) -> Vec<u8> {
    let mut w = HeaderWriter {
        buf: Vec::with_capacity(512),
    };
    w.raw(b"CDF");
    w.buf.push(5);
    w.i64(0); // numrecs: no record variables

    if dims.is_empty() {
        w.tag(0);
        w.i64(0);
    } else {
        w.tag(NC_DIMENSION);
        w.i64(dims.len() as i64);
        for dim in dims {
            w.name(&dim.name);
            w.i64(dim.len as i64);
        }
    }

    w.att_list(gatts);

    if vars.is_empty() {
        w.tag(0);
        w.i64(0);
    } else {
        w.tag(NC_VARIABLE);
        w.i64(vars.len() as i64);
        for var in vars {
            w.name(&var.name);
            w.i64(var.dimids.len() as i64); // rank (CDF-5: 8 bytes)
            for &d in &var.dimids {
                w.i64(d as i64); // dimid (CDF-5: 8 bytes)
            }
            w.att_list(&var.atts);
            w.tag(var.xtype);
            w.i64(var.vsize as i64);
            w.i64(var.begin as i64);
        }
    }
    w.buf
}

/// Assign `vsize` / `begin` to every variable and return the encoded header.
///
/// `vars` is consumed with `begin` and `vsize` unset; the returned header places
/// variable data contiguously right after it.
pub(super) fn finalize(dims: &[DimSpec], gatts: &[Attribute], vars: &mut [VarSpec]) -> Vec<u8> {
    for var in vars.iter_mut() {
        var.vsize = pad4(var.elements(dims) * type_size(var.xtype) as u64);
    }
    // The header size does not depend on the offset values, only their (fixed)
    // width, so a first pass with zeroed offsets gives the true length.
    let header_len = encode_header(dims, gatts, vars).len() as u64;
    let mut offset = header_len;
    for var in vars.iter_mut() {
        var.begin = offset;
        offset += var.vsize;
    }
    encode_header(dims, gatts, vars)
}

// ------------------------------------------------------------------------------
// Reading (CDF-1 / CDF-2 / CDF-5)
// ------------------------------------------------------------------------------

/// Integer widths that vary between classic-format variants.
#[derive(Clone, Copy)]
struct Widths {
    /// `numrecs`, all list counts, name lengths, dimension lengths, attribute
    /// element counts, and `vsize`.
    count: usize,
    /// Variable data offset (`begin`).
    offset: usize,
    /// Dimension-id references inside a variable.
    dimid: usize,
}

pub(super) struct Parsed {
    pub dims: Vec<DimSpec>,
    #[allow(dead_code)]
    pub gatts: Vec<Attribute>,
    pub vars: Vec<VarSpec>,
}

struct HeaderReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    w: Widths,
}

impl<'a> HeaderReader<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        s
    }
    fn i32(&mut self) -> i32 {
        i32::from_be_bytes(self.take(4).try_into().unwrap())
    }
    /// A `count`-width non-negative integer.
    fn count(&mut self) -> u64 {
        match self.w.count {
            4 => u32::from_be_bytes(self.take(4).try_into().unwrap()) as u64,
            8 => u64::from_be_bytes(self.take(8).try_into().unwrap()),
            _ => unreachable!(),
        }
    }
    fn offset(&mut self) -> u64 {
        match self.w.offset {
            4 => u32::from_be_bytes(self.take(4).try_into().unwrap()) as u64,
            8 => u64::from_be_bytes(self.take(8).try_into().unwrap()),
            _ => unreachable!(),
        }
    }
    fn dimid(&mut self) -> usize {
        match self.w.dimid {
            4 => u32::from_be_bytes(self.take(4).try_into().unwrap()) as usize,
            8 => u64::from_be_bytes(self.take(8).try_into().unwrap()) as usize,
            _ => unreachable!(),
        }
    }
    fn align(&mut self) {
        self.pos = self.pos.div_ceil(4) * 4;
    }
    fn name(&mut self) -> String {
        let n = self.count() as usize;
        let s = String::from_utf8_lossy(self.take(n)).into_owned();
        self.align();
        s
    }
    fn att_list(&mut self) -> Vec<Attribute> {
        let tag = self.i32();
        let n = self.count() as usize;
        if tag == 0 {
            return Vec::new();
        }
        assert_eq!(tag, NC_ATTRIBUTE, "expected NC_ATTRIBUTE tag");
        let mut atts = Vec::with_capacity(n);
        for _ in 0..n {
            let name = self.name();
            let xtype = self.i32();
            let len = self.count() as usize;
            let value = match xtype {
                NC_CHAR_TYPE => {
                    let s = String::from_utf8_lossy(self.take(len)).into_owned();
                    AttValue::Text(s)
                }
                NC_INT => AttValue::Int(
                    (0..len)
                        .map(|_| i32::from_be_bytes(self.take(4).try_into().unwrap()))
                        .collect(),
                ),
                NC_FLOAT => AttValue::Float(
                    (0..len)
                        .map(|_| {
                            f32::from_bits(u32::from_be_bytes(self.take(4).try_into().unwrap()))
                        })
                        .collect(),
                ),
                NC_DOUBLE => AttValue::Float(
                    (0..len)
                        .map(|_| {
                            f64::from_bits(u64::from_be_bytes(self.take(8).try_into().unwrap()))
                                as f32
                        })
                        .collect(),
                ),
                other => panic!("unsupported attribute type {other}"),
            };
            self.align();
            atts.push(Attribute { name, value });
        }
        atts
    }
}

pub(super) fn parse(bytes: &[u8]) -> Parsed {
    assert!(bytes.len() >= 4, "file too short to be netCDF");
    assert_eq!(&bytes[..3], b"CDF", "not a netCDF classic file (bad magic)");
    let w = match bytes[3] {
        1 => Widths {
            count: 4,
            offset: 4,
            dimid: 4,
        },
        2 => Widths {
            count: 4,
            offset: 8,
            dimid: 4,
        },
        5 => Widths {
            count: 8,
            offset: 8,
            dimid: 8,
        },
        v => panic!("unsupported netCDF classic version {v}"),
    };
    let mut r = HeaderReader { bytes, pos: 4, w };
    let _numrecs = r.count();

    let dim_tag = r.i32();
    let ndims = r.count() as usize;
    let mut dims = Vec::with_capacity(ndims);
    if dim_tag != 0 {
        assert_eq!(dim_tag, NC_DIMENSION, "expected NC_DIMENSION tag");
        for _ in 0..ndims {
            let name = r.name();
            let len = r.count();
            dims.push(DimSpec { name, len });
        }
    }

    let gatts = r.att_list();

    let var_tag = r.i32();
    let nvars = r.count() as usize;
    let mut vars = Vec::with_capacity(nvars);
    if var_tag != 0 {
        assert_eq!(var_tag, NC_VARIABLE, "expected NC_VARIABLE tag");
        for _ in 0..nvars {
            let name = r.name();
            let rank = r.count() as usize;
            let dimids = (0..rank).map(|_| r.dimid()).collect();
            let atts = r.att_list();
            let xtype = r.i32();
            let vsize = r.count();
            let begin = r.offset();
            vars.push(VarSpec {
                name,
                xtype,
                dimids,
                atts,
                begin,
                vsize,
            });
        }
    }

    Parsed { dims, gatts, vars }
}

// ------------------------------------------------------------------------------
// Data encode / decode (XDR big-endian), fused with the buffer copy
// ------------------------------------------------------------------------------

/// Reinterpret `[T]` as bytes. Sound for every [`NcType`] by that trait's
/// safety contract (`SIZE == size_of::<T>()`, no padding, all bit patterns valid).
fn as_bytes<T: crate::io::netcdf::NcType>(data: &[T]) -> &[u8] {
    // SAFETY: see the [`NcType`] contract; `u8` alignment (1) is always satisfied.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * T::SIZE) }
}

/// Swap `src` into `dst` as `W`-byte big-endian words (`dst.len() == src.len()`,
/// both multiples of `size_of::<W>()`). A no-op copy on a big-endian host.
macro_rules! swap_into {
    ($word:ty, $src:expr, $dst:expr) => {
        for (s, d) in $src
            .chunks_exact(size_of::<$word>())
            .zip($dst.chunks_exact_mut(size_of::<$word>()))
        {
            let word = <$word>::from_ne_bytes(s.try_into().unwrap());
            d.copy_from_slice(&word.to_be_bytes());
        }
    };
}

/// Append `data` to `out` as big-endian XDR bytes.
///
/// The endianness swap happens *inside* the single copy from `data` into `out`
/// (no separate pass), and the destination bytes are made live with `set_len`
/// and fully written before being read (no zero-fill).
pub(super) fn encode_be<T: crate::io::netcdf::NcType>(data: &[T], out: &mut Vec<u8>) {
    let src = as_bytes(data);
    out.reserve(src.len());
    let start = out.len();
    // SAFETY: `reserve` guaranteed the capacity and the match below writes every
    // byte of `start..start + src.len()` before anything reads it.
    unsafe { out.set_len(start + src.len()) }
    let dst = &mut out[start..];
    match T::SIZE {
        4 => swap_into!(u32, src, dst),
        8 => swap_into!(u64, src, dst),
        _ => unreachable!("NcType::SIZE is 4 or 8"),
    }
}

/// Decode `bytes` (big-endian XDR, length a multiple of `T::SIZE`) into `Vec<T>`.
pub(super) fn decode_be<T: crate::io::netcdf::NcType>(bytes: &[u8]) -> Vec<T> {
    let count = bytes.len() / T::SIZE;
    let mut out: Vec<T> = Vec::with_capacity(count);
    // SAFETY: capacity for `count` elements just reserved; the match writes every
    // byte before `set_len` exposes them, and `u8` alignment is trivially met.
    let dst =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), count * T::SIZE) };
    match T::SIZE {
        4 => swap_into!(u32, bytes, dst),
        8 => swap_into!(u64, bytes, dst),
        _ => unreachable!("NcType::SIZE is 4 or 8"),
    }
    // SAFETY: all `count` elements were initialized above.
    unsafe { out.set_len(count) }
    out
}
