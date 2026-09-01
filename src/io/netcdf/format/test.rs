use super::{AttValue, Attribute, DimSpec, VarSpec, finalize, parse, type_size};

fn double_var(name: &str, dimids: Vec<usize>) -> VarSpec {
    VarSpec {
        name: name.to_string(),
        xtype: super::NC_DOUBLE,
        dimids,
        atts: Vec::new(),
        begin: 0,
        vsize: 0,
    }
}

#[test]
fn header_layout_is_contiguous_after_itself() {
    let dims = vec![DimSpec {
        name: "nodes".to_string(),
        len: 4,
    }];
    let gatts = vec![Attribute {
        name: "title".to_string(),
        value: AttValue::Text("t".to_string()),
    }];
    let mut vars = vec![
        double_var("coordx", vec![0]),
        VarSpec {
            name: "connect".to_string(),
            xtype: super::NC_INT,
            dimids: vec![0],
            atts: vec![Attribute {
                name: "elem_type".to_string(),
                value: AttValue::Text("hex8".to_string()),
            }],
            begin: 0,
            vsize: 0,
        },
    ];
    let header = finalize(&dims, &gatts, &mut vars);
    assert_eq!(&header[..4], b"CDF\x05");
    assert_eq!(vars[0].begin, header.len() as u64);
    assert_eq!(vars[0].vsize, 4 * 8);
    assert_eq!(vars[1].begin, vars[0].begin + vars[0].vsize);
    assert_eq!(vars[1].vsize, 4 * 4);
}

#[test]
fn round_trips_through_parse() {
    let dims = vec![
        DimSpec {
            name: "nodes".to_string(),
            len: 3,
        },
        DimSpec {
            name: "time_step".to_string(),
            len: 0,
        },
    ];
    let gatts = vec![
        Attribute {
            name: "file_size".to_string(),
            value: AttValue::Int(vec![1]),
        },
        Attribute {
            name: "version".to_string(),
            value: AttValue::Float(vec![8.25]),
        },
    ];
    let mut vars = vec![double_var("coordx", vec![0])];
    let header = finalize(&dims, &gatts, &mut vars);

    let parsed = parse(&header);
    assert_eq!(parsed.dims.len(), 2);
    assert_eq!(parsed.dims[0].name, "nodes");
    assert_eq!(parsed.dims[0].len, 3);
    assert_eq!(parsed.dims[1].len, 0);
    assert_eq!(parsed.vars.len(), 1);
    assert_eq!(parsed.vars[0].name, "coordx");
    assert_eq!(parsed.vars[0].xtype, super::NC_DOUBLE);
    assert_eq!(parsed.vars[0].begin, vars[0].begin);
    assert_eq!(parsed.vars[0].elements(&parsed.dims), 3);
}

#[test]
fn type_sizes() {
    assert_eq!(type_size(super::NC_INT), 4);
    assert_eq!(type_size(super::NC_FLOAT), 4);
    assert_eq!(type_size(super::NC_DOUBLE), 8);
}

#[test]
#[should_panic(expected = "unsupported netCDF external type")]
fn finalize_rejects_unknown_type() {
    let mut vars = vec![VarSpec {
        name: "x".to_string(),
        xtype: 99,
        dimids: vec![],
        atts: Vec::new(),
        begin: 0,
        vsize: 0,
    }];
    finalize(&[], &[], &mut vars);
}

#[test]
fn empty_dims_and_vars_round_trip() {
    let header = finalize(&[], &[], &mut []);
    let parsed = parse(&header);
    assert!(parsed.dims.is_empty());
    assert!(parsed.gatts.is_empty());
    assert!(parsed.vars.is_empty());
}

struct Buf(Vec<u8>);
impl Buf {
    fn u32(&mut self, x: u32) {
        self.0.extend_from_slice(&x.to_be_bytes());
    }
    fn uint(&mut self, width: usize, x: u64) {
        if width == 8 {
            self.0.extend_from_slice(&x.to_be_bytes());
        } else {
            self.u32(x as u32);
        }
    }
    fn raw(&mut self, b: &[u8]) {
        self.0.extend_from_slice(b);
    }
    fn name(&mut self, count_width: usize, s: &str) {
        self.uint(count_width, s.len() as u64);
        self.raw(s.as_bytes());
        while !self.0.len().is_multiple_of(4) {
            self.0.push(0);
        }
    }
}

fn classic(version: u8, values: &[f64], var_attrs: &[(&str, i32, Vec<u8>)]) -> Vec<u8> {
    let cw = if version == 5 { 8 } else { 4 };
    let ow = if version >= 2 { 8 } else { 4 };
    let dw = if version == 5 { 8 } else { 4 };
    let mut b = Buf(Vec::new());
    b.raw(b"CDF");
    b.0.push(version);
    b.uint(cw, 0);
    b.u32(0x0A);
    b.uint(cw, 1);
    b.name(cw, "n");
    b.uint(cw, values.len() as u64);
    b.u32(0);
    b.uint(cw, 0);
    b.u32(0x0B);
    b.uint(cw, 1);
    b.name(cw, "x");
    b.uint(cw, 1);
    b.uint(dw, 0);
    if var_attrs.is_empty() {
        b.u32(0);
        b.uint(cw, 0);
    } else {
        b.u32(0x0C);
        b.uint(cw, var_attrs.len() as u64);
        for (name, tag, payload) in var_attrs {
            let nelems = match tag {
                2 => payload.len(),
                6 => payload.len() / 8,
                _ => payload.len() / 4,
            };
            b.name(cw, name);
            b.u32(*tag as u32);
            b.uint(cw, nelems as u64);
            b.raw(payload);
            while !b.0.len().is_multiple_of(4) {
                b.0.push(0);
            }
        }
    }
    b.u32(6);
    b.uint(cw, (values.len() * 8) as u64);
    let begin = b.0.len() as u64 + ow as u64;
    b.uint(ow, begin);
    for &v in values {
        b.raw(&v.to_bits().to_be_bytes());
    }
    b.0
}

#[test]
fn parses_cdf1_and_cdf2() {
    for version in [1u8, 2] {
        let values = [1.5_f64, -2.25, 42.0];
        let bytes = classic(version, &values, &[]);
        let parsed = parse(&bytes);
        assert_eq!(parsed.dims[0].name, "n");
        assert_eq!(parsed.dims[0].len, 3);
        let var = &parsed.vars[0];
        assert_eq!(var.name, "x");
        assert_eq!(var.xtype, super::NC_DOUBLE);
        assert_eq!(var.elements(&parsed.dims), 3);
        let start = var.begin as usize;
        let read: Vec<f64> = super::decode_be(&bytes[start..start + 24]);
        assert_eq!(read, values);
    }
}

#[test]
fn reads_int_and_double_variable_attributes() {
    let attrs: [(&str, i32, Vec<u8>); 3] = [
        ("tag", 2, b"hi".to_vec()),
        ("count", 4, 7_i32.to_be_bytes().to_vec()),
        ("scale", 6, 2.5_f64.to_bits().to_be_bytes().to_vec()),
    ];
    let bytes = classic(5, &[0.0], &attrs);
    let parsed = parse(&bytes);
    let var = &parsed.vars[0];
    assert!(matches!(&var.atts[0].value, AttValue::Text(s) if s == "hi"));
    assert!(matches!(&var.atts[1].value, AttValue::Int(v) if v == &[7]));
    assert!(matches!(&var.atts[2].value, AttValue::Float(v) if v == &[2.5]));
}

#[test]
#[should_panic(expected = "unsupported attribute type")]
fn rejects_unknown_attribute_type() {
    let attrs = [("weird", 99_i32, vec![0, 0, 0, 0])];
    parse(&classic(5, &[0.0], &attrs));
}

#[test]
#[should_panic(expected = "bad magic")]
fn rejects_bad_magic() {
    parse(b"NOPE\x00\x00\x00\x00");
}

#[test]
#[should_panic(expected = "unsupported netCDF classic version")]
fn rejects_unknown_version() {
    parse(b"CDF\x09\x00\x00\x00\x00");
}
