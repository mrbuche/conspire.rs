use super::{
    AttValue, Attribute, DimSpec, NC_DOUBLE, NC_FLOAT, NC_INT, VarSpec, finalize, parse, type_size,
};

fn double_var(name: &str, dimids: Vec<usize>) -> VarSpec {
    VarSpec {
        name: name.to_string(),
        xtype: NC_DOUBLE,
        dimids,
        atts: Vec::new(),
        begin: 0,
        vsize: 0,
    }
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
    assert_eq!(parsed.vars[0].xtype, NC_DOUBLE);
    assert_eq!(parsed.vars[0].begin, vars[0].begin);
    assert_eq!(parsed.vars[0].elements(&parsed.dims), 3);
}

#[test]
fn type_sizes() {
    assert_eq!(type_size(NC_INT), 4);
    assert_eq!(type_size(NC_FLOAT), 4);
    assert_eq!(type_size(NC_DOUBLE), 8);
}

#[test]
fn empty_dims_and_vars_round_trip() {
    let header = finalize(&[], &[], &mut []);
    let parsed = parse(&header);
    assert!(parsed.dims.is_empty());
    assert!(parsed.gatts.is_empty());
    assert!(parsed.vars.is_empty());
}
