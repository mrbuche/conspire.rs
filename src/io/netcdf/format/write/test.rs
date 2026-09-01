use super::finalize;
use crate::io::netcdf::format::{
    AttValue, Attribute, DimSpec, NC_DOUBLE, NC_INT, Storage, VarSpec,
};

fn double_var(name: &str, dimids: Vec<usize>) -> VarSpec {
    VarSpec {
        name: name.to_string(),
        xtype: NC_DOUBLE,
        dimids,
        atts: Vec::new(),
        begin: 0,
        vsize: 0,
        storage: Storage::Classic,
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
            xtype: NC_INT,
            dimids: vec![0],
            atts: vec![Attribute {
                name: "elem_type".to_string(),
                value: AttValue::Text("hex8".to_string()),
            }],
            begin: 0,
            vsize: 0,
            storage: Storage::Classic,
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
#[should_panic(expected = "unsupported netCDF external type")]
fn finalize_rejects_unknown_type() {
    let mut vars = vec![VarSpec {
        name: "x".to_string(),
        xtype: 99,
        dimids: vec![],
        atts: Vec::new(),
        begin: 0,
        vsize: 0,
        storage: Storage::Classic,
    }];
    finalize(&[], &[], &mut vars);
}
