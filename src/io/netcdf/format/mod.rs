//! Minimal reader and writer for the netCDF *classic* formats (CDF-1, CDF-2,
//! CDF-5), sufficient for Exodus mesh files: fixed-size (non-record) variables
//! only, whole-variable access, `int` / `float` / `double` data.
//!
//! Files are always written as CDF-5 ("64-bit data"); all three classic
//! variants are accepted on read. NetCDF-4 (HDF5) files are not supported.
//!
//! The work splits three ways: [`write`] encodes a CDF-5 header, [`read`] parses
//! any classic header, and [`xdr`] moves variable data between native order and
//! netCDF's big-endian encoding. This module holds the vocabulary they share.

#[cfg(test)]
mod test;

mod read;
mod write;
mod xdr;

pub(super) use read::parse;
pub(super) use write::finalize;
pub(super) use xdr::{decode_be, encode_be};

pub(super) const NC_INT: i32 = 4;
pub(super) const NC_FLOAT: i32 = 5;
pub(super) const NC_DOUBLE: i32 = 6;

const NC_CHAR_TYPE: i32 = 2;
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

pub(super) struct Parsed {
    pub dims: Vec<DimSpec>,
    #[allow(dead_code)]
    pub gatts: Vec<Attribute>,
    pub vars: Vec<VarSpec>,
}
