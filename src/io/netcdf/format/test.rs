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
