use super::{Exchange, Parameter, parse};

const CUBE: &str = r#"
ISO-10303-21;
HEADER;
/* a planar solid */
FILE_DESCRIPTION(('a unit cube'),'2;1');
FILE_NAME('cube.step','2026-08-27T00:00:00',('mrbuche'),(''),'conspire','conspire','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 }'));
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('',(0.,0.,0.));
#2 = CARTESIAN_POINT('',(1.,1.,1.E0));
#3 = DIRECTION('',(0.,0.,1.));
#4 = DIRECTION('',(1.,0.,0.));
#5 = AXIS2_PLACEMENT_3D('',#1,#3,#4);
#6 = PLANE('',#5);
#7 = VERTEX_POINT('',#2);
#8 = ( NAMED_UNIT(*) SI_UNIT($,.METRE.) LENGTH_UNIT() );
#9 = MEASURE_WITH_UNIT(LENGTH_MEASURE(1.),#8);
#10 = MANIFOLD_SOLID_BREP('cube',#6);
ENDSEC;
END-ISO-10303-21;
"#;

#[test]
fn parses_structure() {
    let exchange: Exchange = CUBE.parse().unwrap();
    assert_eq!(exchange.header.len(), 3);
    assert_eq!(exchange.header[0].keyword, "FILE_DESCRIPTION");
    assert_eq!(exchange.data.len(), 10);
    assert_eq!(exchange.data[&10].records[0].keyword, "MANIFOLD_SOLID_BREP");
}

#[test]
fn parses_parameter_kinds() {
    let exchange = parse(CUBE).unwrap();
    let brep = &exchange.data[&10].records[0].parameters;
    assert_eq!(brep[0], Parameter::String("cube".into()));
    assert_eq!(brep[1], Parameter::Reference(6));
    let point = &exchange.data[&2].records[0].parameters;
    assert_eq!(point[0], Parameter::String(String::new()));
    assert_eq!(
        point[1],
        Parameter::List(vec![
            Parameter::Real(1.0),
            Parameter::Real(1.0),
            Parameter::Real(1.0),
        ])
    );
}

#[test]
fn parses_typed_and_null_parameters() {
    let exchange = parse(CUBE).unwrap();
    let unit = &exchange.data[&8];
    assert_eq!(unit.records.len(), 3);
    assert_eq!(unit.records[0].parameters, vec![Parameter::Derived]);
    assert_eq!(
        unit.records[1].parameters,
        vec![Parameter::Null, Parameter::Enumeration("METRE".into())]
    );
    let measure = &exchange.data[&9].records[0].parameters;
    assert_eq!(
        measure[0],
        Parameter::Typed {
            keyword: "LENGTH_MEASURE".into(),
            parameter: Box::new(Parameter::Real(1.0)),
        }
    );
}

#[test]
fn parses_integers_and_signed_exponents() {
    let text = wrap("#1 = ITEM(3,-4,+5.0,-2.5E-3,6.e+2);");
    let parameters = &parse(&text).unwrap().data[&1].records[0].parameters;
    assert_eq!(
        parameters,
        &[
            Parameter::Integer(3),
            Parameter::Integer(-4),
            Parameter::Real(5.0),
            Parameter::Real(-2.5e-3),
            Parameter::Real(6.0e2),
        ]
    );
}

#[test]
fn unescapes_doubled_apostrophe() {
    let text = wrap("#1 = LABEL('o''brien''s part');");
    let parameters = &parse(&text).unwrap().data[&1].records[0].parameters;
    assert_eq!(parameters[0], Parameter::String("o'brien's part".into()));
}

#[test]
fn rejects_undefined_reference() {
    let text = wrap("#1 = A(#2);");
    assert!(parse(&text).unwrap_err().to_string().contains("#2"));
}

#[test]
fn rejects_duplicate_id() {
    let text = wrap("#1 = A();\n#1 = B();");
    assert!(parse(&text).unwrap_err().to_string().contains("duplicate"));
}

#[test]
fn rejects_missing_terminator() {
    let text = wrap("#1 = A()\n#2 = B();");
    assert!(parse(&text).is_err());
}

#[test]
fn rejects_truncated_file() {
    assert!(parse("ISO-10303-21;\nHEADER;\n").is_err());
}

#[test]
fn rejects_pathologically_nested_parameters() {
    // Without a depth cap this recursion overflows the stack (an uncatchable
    // abort); it must come back as an ordinary Err.
    let deep = format!("#1 = A({}{});", "(".repeat(5000), ")".repeat(5000));
    let error = parse(&wrap(&deep)).unwrap_err().to_string();
    assert!(error.contains("depth limit"), "{error}");
}

#[test]
fn rejects_a_non_finite_real() {
    // Overflow -> +/-inf and underflow -> exactly 0.0 both parse `Ok` from
    // `str::parse`; neither may reach geometry.
    for literal in ["1.0E400", "1.0E-400"] {
        let error = parse(&wrap(&format!("#1 = P({literal});")))
            .unwrap_err()
            .to_string();
        assert!(error.contains("out-of-range real"), "{literal}: {error}");
    }
    // A genuine zero and a genuine tiny value still parse.
    assert!(parse(&wrap("#1 = P(0.,-0.0,1.0E-30);")).is_ok());
}

#[test]
fn tolerates_a_leading_bom() {
    let text = format!("\u{feff}{}", wrap("#1 = A();"));
    assert!(parse(&text).is_ok());
    // The 3-byte BOM is added back into reported offsets, so an error points
    // at the real position in the file the caller passed, not the stripped one.
    let bare = parse(&wrap("#1 = A(@);")).unwrap_err().to_string();
    let with_bom = parse(&format!("\u{feff}{}", wrap("#1 = A(@);")))
        .unwrap_err()
        .to_string();
    let offset = |message: &str| {
        message
            .rsplit(' ')
            .next()
            .unwrap()
            .parse::<usize>()
            .unwrap()
    };
    assert_eq!(offset(&with_bom), offset(&bare) + 3);
}

#[test]
fn rejects_trailing_content() {
    let mut text = wrap("#1 = A();");
    text.push_str("STRAY DATA");
    assert!(
        parse(&text)
            .unwrap_err()
            .to_string()
            .contains("trailing content")
    );
}

#[test]
fn rejects_an_unterminated_comment() {
    // Trailing `/*` used to be swallowed to EOF, silently defeating the
    // end-of-file check.
    let text = format!("{}/* dangling", wrap("#1 = A();"));
    assert!(
        parse(&text)
            .unwrap_err()
            .to_string()
            .contains("unterminated"),
        "{}",
        parse(&text).unwrap_err()
    );
}

#[test]
fn rejects_a_header_entity_reference() {
    let text = "ISO-10303-21;\nHEADER;\nFILE_NAME(#5);\nENDSEC;\nDATA;\n#1 = A();\nENDSEC;\nEND-ISO-10303-21;\n";
    assert!(parse(text).unwrap_err().to_string().contains("HEADER"));
}

#[test]
fn rejects_control_characters_in_a_string() {
    let error = parse(&wrap("#1 = L('a\nb');")).unwrap_err().to_string();
    assert!(error.contains("control character"), "{error}");
}

#[test]
fn rejects_a_leading_dot_real() {
    // `.5` reaches `enumeration`; `-.5` reaches `number` -- both malformed.
    assert!(parse(&wrap("#1 = A(-.5);")).is_err());
    assert!(parse(&wrap("#1 = A(.5);")).is_err());
    // The well-formed forms still parse.
    let params = &parse(&wrap("#1 = A(-0.5,5.);")).unwrap().data[&1].records[0].parameters;
    assert_eq!(params, &[Parameter::Real(-0.5), Parameter::Real(5.0)]);
}

#[test]
fn rejects_a_numeric_enumeration() {
    assert!(parse(&wrap("#1 = A(.123.);")).is_err());
    assert!(parse(&wrap("#1 = A(.T.,.STEEL_2.);")).is_ok());
}

#[test]
fn rejects_malformed_entity_ids() {
    assert!(parse(&wrap("#0 = A();")).is_err());
    assert!(parse(&wrap("#007 = A();")).is_err());
    assert!(parse(&wrap("#1 = A(#02);")).is_err());
}

fn wrap(data: &str) -> String {
    format!("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{data}\nENDSEC;\nEND-ISO-10303-21;\n")
}
