#[cfg(test)]
mod test;

pub(super) mod hdf5;
mod read;
mod write;
mod xdr;

pub(super) use write::finalize;
pub(super) use xdr::{decode_be, decode_le, encode_be, encode_le};

pub(super) fn parse(bytes: &[u8]) -> Parsed {
    match hdf5::superblock_offset(bytes) {
        Some(base) => hdf5::parse(bytes, base),
        None => read::parse(bytes),
    }
}

pub(super) const NC_INT: i32 = 4;
pub(super) const NC_FLOAT: i32 = 5;
pub(super) const NC_DOUBLE: i32 = 6;

const NC_CHAR_TYPE: i32 = 2;
const NC_DIMENSION: i32 = 0x0A;
const NC_VARIABLE: i32 = 0x0B;
const NC_ATTRIBUTE: i32 = 0x0C;

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
    pub begin: u64,
    pub vsize: u64,
    pub storage: Storage,
}

pub(super) enum Storage {
    Classic,
    Hdf5 {
        little_endian: bool,
        layout: hdf5::Layout,
        filters: Vec<hdf5::Filter>,
    },
}

impl VarSpec {
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
