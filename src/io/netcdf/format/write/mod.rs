#[cfg(test)]
mod test;

use super::{
    AttValue, Attribute, DimSpec, NC_ATTRIBUTE, NC_CHAR_TYPE, NC_DIMENSION, NC_FLOAT, NC_INT,
    NC_VARIABLE, VarSpec, type_size,
};

fn pad4(n: u64) -> u64 {
    n.div_ceil(4) * 4
}

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
        self.i64(s.len() as i64);
        self.raw(s.as_bytes());
        self.align();
    }
    fn att_list(&mut self, atts: &[Attribute]) {
        if atts.is_empty() {
            self.tag(0);
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

fn encode_header(dims: &[DimSpec], gatts: &[Attribute], vars: &[VarSpec]) -> Vec<u8> {
    let mut w = HeaderWriter {
        buf: Vec::with_capacity(512),
    };
    w.raw(b"CDF");
    w.buf.push(5);
    w.i64(0);

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
            w.i64(var.dimids.len() as i64);
            for &d in &var.dimids {
                w.i64(d as i64);
            }
            w.att_list(&var.atts);
            w.tag(var.xtype);
            w.i64(var.vsize as i64);
            w.i64(var.begin as i64);
        }
    }
    w.buf
}

pub(in crate::io::netcdf) fn finalize(
    dims: &[DimSpec],
    gatts: &[Attribute],
    vars: &mut [VarSpec],
) -> Vec<u8> {
    for var in vars.iter_mut() {
        var.vsize = pad4(var.elements(dims) * type_size(var.xtype) as u64);
    }
    let header_len = encode_header(dims, gatts, vars).len() as u64;
    let mut offset = header_len;
    for var in vars.iter_mut() {
        var.begin = offset;
        offset += var.vsize;
    }
    encode_header(dims, gatts, vars)
}
