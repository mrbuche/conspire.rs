//! Classic-format header parsing (CDF-1, CDF-2, and CDF-5).

#[cfg(test)]
mod test;

use super::{
    AttValue, Attribute, DimSpec, NC_ATTRIBUTE, NC_CHAR_TYPE, NC_DIMENSION, NC_DOUBLE, NC_FLOAT,
    NC_INT, NC_VARIABLE, Parsed, VarSpec,
};

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
    /// A big-endian non-negative integer of `width` bytes (4 or 8).
    fn uint(&mut self, width: usize) -> u64 {
        if width == 8 {
            u64::from_be_bytes(self.take(8).try_into().unwrap())
        } else {
            u32::from_be_bytes(self.take(4).try_into().unwrap()) as u64
        }
    }
    /// A `count`-width non-negative integer (list lengths, sizes, `numrecs`).
    fn count(&mut self) -> u64 {
        self.uint(self.w.count)
    }
    /// The variable data offset (`begin`).
    fn offset(&mut self) -> u64 {
        self.uint(self.w.offset)
    }
    /// A dimension-id reference inside a variable.
    fn dimid(&mut self) -> usize {
        self.uint(self.w.dimid) as usize
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

pub(in crate::io::netcdf) fn parse(bytes: &[u8]) -> Parsed {
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
