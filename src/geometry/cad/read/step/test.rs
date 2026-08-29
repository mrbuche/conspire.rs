use super::read;

const CUBE: &str = r#"
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('unit cube'),'2;1');
FILE_NAME('cube.step','2026-08-27T00:00:00',(''),(''),'conspire','conspire','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 }'));
ENDSEC;
DATA;
#10 = CARTESIAN_POINT('',(0.,0.,0.));
#11 = CARTESIAN_POINT('',(1.,0.,0.));
#12 = CARTESIAN_POINT('',(1.,1.,0.));
#13 = CARTESIAN_POINT('',(0.,1.,0.));
#14 = CARTESIAN_POINT('',(0.,0.,1.));
#15 = CARTESIAN_POINT('',(1.,0.,1.));
#16 = CARTESIAN_POINT('',(1.,1.,1.));
#17 = CARTESIAN_POINT('',(0.,1.,1.));
#20 = DIRECTION('',(0.,0.,1.));
#21 = DIRECTION('',(0.,0.,-1.));
#22 = DIRECTION('',(1.,0.,0.));
#23 = DIRECTION('',(-1.,0.,0.));
#24 = DIRECTION('',(0.,1.,0.));
#25 = DIRECTION('',(0.,-1.,0.));
#30 = VERTEX_POINT('',#10);
#31 = VERTEX_POINT('',#11);
#32 = VERTEX_POINT('',#12);
#33 = VERTEX_POINT('',#13);
#34 = VERTEX_POINT('',#14);
#35 = VERTEX_POINT('',#15);
#36 = VERTEX_POINT('',#16);
#37 = VERTEX_POINT('',#17);
#40 = VECTOR('',#22,1.);
#41 = VECTOR('',#24,1.);
#42 = VECTOR('',#23,1.);
#43 = VECTOR('',#25,1.);
#44 = VECTOR('',#20,1.);
#50 = LINE('',#10,#40);
#51 = LINE('',#11,#41);
#52 = LINE('',#12,#42);
#53 = LINE('',#13,#43);
#54 = LINE('',#14,#40);
#55 = LINE('',#15,#41);
#56 = LINE('',#16,#42);
#57 = LINE('',#17,#43);
#58 = LINE('',#10,#44);
#59 = LINE('',#11,#44);
#60 = LINE('',#12,#44);
#61 = LINE('',#13,#44);
#70 = EDGE_CURVE('',#30,#31,#50,.T.);
#71 = EDGE_CURVE('',#31,#32,#51,.T.);
#72 = EDGE_CURVE('',#32,#33,#52,.T.);
#73 = EDGE_CURVE('',#33,#30,#53,.T.);
#74 = EDGE_CURVE('',#34,#35,#54,.T.);
#75 = EDGE_CURVE('',#35,#36,#55,.T.);
#76 = EDGE_CURVE('',#36,#37,#56,.T.);
#77 = EDGE_CURVE('',#37,#34,#57,.T.);
#78 = EDGE_CURVE('',#30,#34,#58,.T.);
#79 = EDGE_CURVE('',#31,#35,#59,.T.);
#80 = EDGE_CURVE('',#32,#36,#60,.T.);
#81 = EDGE_CURVE('',#33,#37,#61,.T.);
#90 = AXIS2_PLACEMENT_3D('',#10,#21,#22);
#91 = AXIS2_PLACEMENT_3D('',#14,#20,#22);
#92 = AXIS2_PLACEMENT_3D('',#10,#25,#22);
#93 = AXIS2_PLACEMENT_3D('',#13,#24,#22);
#94 = AXIS2_PLACEMENT_3D('',#10,#23,#24);
#95 = AXIS2_PLACEMENT_3D('',#11,#22,#24);
#100 = PLANE('',#90);
#101 = PLANE('',#91);
#102 = PLANE('',#92);
#103 = PLANE('',#93);
#104 = PLANE('',#94);
#105 = PLANE('',#95);
#110 = ORIENTED_EDGE('',*,*,#73,.F.);
#111 = ORIENTED_EDGE('',*,*,#72,.F.);
#112 = ORIENTED_EDGE('',*,*,#71,.F.);
#113 = ORIENTED_EDGE('',*,*,#70,.F.);
#114 = ORIENTED_EDGE('',*,*,#74,.T.);
#115 = ORIENTED_EDGE('',*,*,#75,.T.);
#116 = ORIENTED_EDGE('',*,*,#76,.T.);
#117 = ORIENTED_EDGE('',*,*,#77,.T.);
#118 = ORIENTED_EDGE('',*,*,#70,.T.);
#119 = ORIENTED_EDGE('',*,*,#79,.T.);
#120 = ORIENTED_EDGE('',*,*,#74,.F.);
#121 = ORIENTED_EDGE('',*,*,#78,.F.);
#122 = ORIENTED_EDGE('',*,*,#81,.T.);
#123 = ORIENTED_EDGE('',*,*,#76,.F.);
#124 = ORIENTED_EDGE('',*,*,#80,.F.);
#125 = ORIENTED_EDGE('',*,*,#72,.T.);
#126 = ORIENTED_EDGE('',*,*,#78,.T.);
#127 = ORIENTED_EDGE('',*,*,#77,.F.);
#128 = ORIENTED_EDGE('',*,*,#81,.F.);
#129 = ORIENTED_EDGE('',*,*,#73,.T.);
#130 = ORIENTED_EDGE('',*,*,#71,.T.);
#131 = ORIENTED_EDGE('',*,*,#80,.T.);
#132 = ORIENTED_EDGE('',*,*,#75,.F.);
#133 = ORIENTED_EDGE('',*,*,#79,.F.);
#140 = EDGE_LOOP('',(#110,#111,#112,#113));
#141 = EDGE_LOOP('',(#114,#115,#116,#117));
#142 = EDGE_LOOP('',(#118,#119,#120,#121));
#143 = EDGE_LOOP('',(#122,#123,#124,#125));
#144 = EDGE_LOOP('',(#126,#127,#128,#129));
#145 = EDGE_LOOP('',(#130,#131,#132,#133));
#150 = FACE_OUTER_BOUND('',#140,.T.);
#151 = FACE_OUTER_BOUND('',#141,.T.);
#152 = FACE_OUTER_BOUND('',#142,.T.);
#153 = FACE_OUTER_BOUND('',#143,.T.);
#154 = FACE_OUTER_BOUND('',#144,.T.);
#155 = FACE_OUTER_BOUND('',#145,.T.);
#160 = ADVANCED_FACE('',(#150),#100,.T.);
#161 = ADVANCED_FACE('',(#151),#101,.T.);
#162 = ADVANCED_FACE('',(#152),#102,.T.);
#163 = ADVANCED_FACE('',(#153),#103,.T.);
#164 = ADVANCED_FACE('',(#154),#104,.T.);
#165 = ADVANCED_FACE('',(#155),#105,.T.);
#170 = CLOSED_SHELL('',(#160,#161,#162,#163,#164,#165));
#180 = MANIFOLD_SOLID_BREP('cube',#170);
ENDSEC;
END-ISO-10303-21;
"#;

#[test]
fn reads_cube_topology() {
    let brep = read(CUBE).unwrap();
    assert_eq!(brep.vertices.len(), 8);
    assert_eq!(brep.edges.len(), 12);
    assert_eq!(brep.faces.len(), 6);
    assert_eq!(brep.shells.len(), 1);
    assert!(brep.shells[0].closed);
    assert_eq!(brep.shells[0].faces, (0..6).collect::<Vec<_>>());
}

#[test]
fn tessellates_read_cube() {
    let brep = read(CUBE).unwrap();
    let tessellation = brep.tessellate().unwrap();
    let mesh = tessellation.mesh();
    assert_eq!(mesh.number_of_nodes(), 8);
    let crate::geometry::mesh::Connectivity::Triangular(block) = &mesh.connectivities()[0] else {
        panic!("expected a triangular mesh");
    };
    let triangles: Vec<[usize; 3]> = block.iter().copied().collect();
    assert_eq!(triangles.len(), 12);

    let point = |node: usize| {
        let coordinate = &mesh.coordinates()[node];
        [
            coordinate[0].value(),
            coordinate[1].value(),
            coordinate[2].value(),
        ]
    };
    let mut area = 0.0f64;
    for &[a, b, c] in triangles.iter() {
        let (pa, pb, pc) = (point(a), point(b), point(c));
        let u = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let v = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let normal = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        area += 0.5 * (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
        let centroid = [
            (pa[0] + pb[0] + pc[0]) / 3.0 - 0.5,
            (pa[1] + pb[1] + pc[1]) / 3.0 - 0.5,
            (pa[2] + pb[2] + pc[2]) / 3.0 - 0.5,
        ];
        let outward = normal[0] * centroid[0] + normal[1] * centroid[1] + normal[2] * centroid[2];
        assert!(outward > 0.0, "triangle {:?} winds inward", [a, b, c]);
    }
    assert!((area - 6.0).abs() < 1e-9, "surface area was {area}");
}

/// A capped cylinder: radius 2, height 5, axis +z, base centred at the origin.
/// Two planar disk caps and one cylindrical lateral face split by a seam line at
/// angle 0, so the two circular rim edges share a vertex with the seam.
const CYLINDER: &str = r#"
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('capped cylinder'),'2;1');
FILE_NAME('cylinder.step','2026-08-28T00:00:00',(''),(''),'conspire','conspire','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 }'));
ENDSEC;
DATA;
#10 = CARTESIAN_POINT('',(0.,0.,0.));
#11 = CARTESIAN_POINT('',(0.,0.,5.));
#12 = CARTESIAN_POINT('',(2.,0.,0.));
#13 = CARTESIAN_POINT('',(2.,0.,5.));
#20 = DIRECTION('',(0.,0.,1.));
#21 = DIRECTION('',(0.,0.,-1.));
#22 = DIRECTION('',(1.,0.,0.));
#30 = VERTEX_POINT('',#12);
#31 = VERTEX_POINT('',#13);
#40 = AXIS2_PLACEMENT_3D('',#10,#20,#22);
#41 = AXIS2_PLACEMENT_3D('',#11,#20,#22);
#42 = CIRCLE('',#40,2.);
#43 = CIRCLE('',#41,2.);
#44 = VECTOR('',#20,1.);
#45 = LINE('',#12,#44);
#50 = EDGE_CURVE('',#30,#30,#42,.T.);
#51 = EDGE_CURVE('',#31,#31,#43,.T.);
#52 = EDGE_CURVE('',#30,#31,#45,.T.);
#60 = AXIS2_PLACEMENT_3D('',#10,#21,#22);
#61 = AXIS2_PLACEMENT_3D('',#11,#20,#22);
#62 = PLANE('',#60);
#63 = PLANE('',#61);
#64 = AXIS2_PLACEMENT_3D('',#10,#20,#22);
#65 = CYLINDRICAL_SURFACE('',#64,2.);
#70 = ORIENTED_EDGE('',*,*,#50,.F.);
#71 = ORIENTED_EDGE('',*,*,#51,.T.);
#72 = ORIENTED_EDGE('',*,*,#50,.T.);
#73 = ORIENTED_EDGE('',*,*,#52,.T.);
#74 = ORIENTED_EDGE('',*,*,#51,.F.);
#75 = ORIENTED_EDGE('',*,*,#52,.F.);
#80 = EDGE_LOOP('',(#70));
#81 = EDGE_LOOP('',(#71));
#82 = EDGE_LOOP('',(#72,#73,#74,#75));
#90 = FACE_OUTER_BOUND('',#80,.T.);
#91 = FACE_OUTER_BOUND('',#81,.T.);
#92 = FACE_OUTER_BOUND('',#82,.T.);
#100 = ADVANCED_FACE('',(#90),#62,.T.);
#101 = ADVANCED_FACE('',(#91),#63,.T.);
#102 = ADVANCED_FACE('',(#92),#65,.T.);
#110 = CLOSED_SHELL('',(#100,#101,#102));
#120 = MANIFOLD_SOLID_BREP('cylinder',#110);
ENDSEC;
END-ISO-10303-21;
"#;

#[test]
fn reads_cylinder_topology() {
    use crate::geometry::cad::brep::{curve::Curve, surface::Surface};

    let brep = read(CYLINDER).unwrap();
    assert_eq!(brep.vertices.len(), 2);
    assert_eq!(brep.edges.len(), 3);
    assert_eq!(brep.faces.len(), 3);
    assert_eq!(brep.shells.len(), 1);
    assert!(brep.shells[0].closed);
    assert_eq!(brep.shells[0].faces, vec![0, 1, 2]);

    let planar = brep
        .faces
        .iter()
        .filter(|face| matches!(face.surface, Surface::Plane(_)))
        .count();
    assert_eq!(planar, 2);

    let Surface::Cylinder(cylinder) = &brep.faces[2].surface else {
        panic!("lateral face is not cylindrical");
    };
    assert_eq!(cylinder.radius, 2.0);
    assert_eq!(
        cylinder.axis,
        crate::geometry::Direction::const_from([0.0, 0.0, 1.0])
    );

    let circles = brep
        .edges
        .iter()
        .filter(|edge| matches!(edge.curve, Curve::Circle(_)))
        .count();
    assert_eq!(circles, 2);

    // The seam line joins the two rim vertices; each rim circle closes on one.
    let Curve::Circle(rim) = &brep.edges[0].curve else {
        panic!("edge 0 is not a circle");
    };
    assert_eq!(rim.radius, 2.0);
    assert_eq!(brep.edges[2].vertices, [0, 1]);
}

#[test]
fn read_cylinder_recognised_as_a_primitive_and_meshed() {
    use crate::{
        geometry::{
            Coordinate,
            csg::Primitive,
            mesh::{Connectivity, Fitting, Verdict},
            ntree::Balancing,
            solid::{Solid, SolidOracle, Uniform},
        },
        math::Quantity,
        units::Length,
    };

    let Some(Primitive::Cylinder(cylinder)) = read(CYLINDER).unwrap().primitive() else {
        panic!("read cylinder not recognised as a primitive");
    };
    let oracle = cylinder.oracle().unwrap();
    assert!((oracle.signed_distance(&Coordinate::from([0.0, 0.0, 2.5])) - 2.0).abs() < 1e-9);

    let mesh = cylinder
        .mesh(
            &Uniform(Quantity::<Length>::new(0.6)),
            6,
            0.1,
            Balancing::Strong(1),
            Fitting::Soft,
        )
        .unwrap();
    assert!(matches!(
        mesh.connectivities()[0],
        Connectivity::Hexahedral(_)
    ));
    assert!(mesh.minimum_scaled_jacobians()[0].iter().all(|&j| j > 0.0));

    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for coordinate in mesh.coordinates() {
        for k in 0..3 {
            low[k] = low[k].min(coordinate[k].value());
            high[k] = high[k].max(coordinate[k].value());
        }
    }
    assert!((low[0] + 2.0).abs() < 0.25 && (high[0] - 2.0).abs() < 0.25);
    assert!(low[2].abs() < 0.1 && (high[2] - 5.0).abs() < 0.1);
}

/// A sphere of radius 3 centred at the origin: one periodic `SPHERICAL_SURFACE`
/// face with a meridian seam between the two pole vertices.
const SPHERE: &str = r#"
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('sphere'),'2;1');
FILE_NAME('sphere.step','2026-08-28T00:00:00',(''),(''),'conspire','conspire','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 }'));
ENDSEC;
DATA;
#10 = CARTESIAN_POINT('',(0.,0.,0.));
#11 = CARTESIAN_POINT('',(0.,0.,-3.));
#12 = CARTESIAN_POINT('',(0.,0.,3.));
#20 = DIRECTION('',(0.,0.,1.));
#21 = DIRECTION('',(1.,0.,0.));
#22 = DIRECTION('',(0.,1.,0.));
#30 = VERTEX_POINT('',#11);
#31 = VERTEX_POINT('',#12);
#40 = AXIS2_PLACEMENT_3D('',#10,#22,#21);
#41 = CIRCLE('',#40,3.);
#50 = EDGE_CURVE('',#30,#31,#41,.T.);
#60 = AXIS2_PLACEMENT_3D('',#10,#20,#21);
#61 = SPHERICAL_SURFACE('',#60,3.);
#70 = ORIENTED_EDGE('',*,*,#50,.T.);
#71 = ORIENTED_EDGE('',*,*,#50,.F.);
#80 = EDGE_LOOP('',(#70,#71));
#90 = FACE_OUTER_BOUND('',#80,.T.);
#100 = ADVANCED_FACE('',(#90),#61,.T.);
#110 = CLOSED_SHELL('',(#100));
#120 = MANIFOLD_SOLID_BREP('sphere',#110);
ENDSEC;
END-ISO-10303-21;
"#;

#[test]
fn reads_spherical_surface_and_recognises_the_primitive() {
    use crate::geometry::{Coordinate, cad::brep::surface::Surface, csg::Primitive};

    let brep = read(SPHERE).unwrap();
    assert_eq!(brep.vertices.len(), 2);
    assert_eq!(brep.edges.len(), 1);
    assert_eq!(brep.faces.len(), 1);

    let Surface::Sphere(sphere) = &brep.faces[0].surface else {
        panic!("face is not spherical");
    };
    assert_eq!(sphere.radius, 3.0);
    assert_eq!(sphere.origin, Coordinate::from([0.0, 0.0, 0.0]));

    let Some(Primitive::Sphere(_)) = brep.primitive() else {
        panic!("sphere not recognised as a primitive");
    };
}

/// The same capped cylinder, but the rim and seam edges reference their 3D
/// geometry indirectly through `SEAM_CURVE` / `SURFACE_CURVE` wrappers, the way
/// most kernels actually export trimmed analytic edges.
const CYLINDER_INDIRECT: &str = r#"
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('capped cylinder, indirect edge geometry'),'2;1');
FILE_NAME('cylinder.step','2026-08-28T00:00:00',(''),(''),'conspire','conspire','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 }'));
ENDSEC;
DATA;
#10 = CARTESIAN_POINT('',(0.,0.,0.));
#11 = CARTESIAN_POINT('',(0.,0.,5.));
#12 = CARTESIAN_POINT('',(2.,0.,0.));
#13 = CARTESIAN_POINT('',(2.,0.,5.));
#20 = DIRECTION('',(0.,0.,1.));
#21 = DIRECTION('',(0.,0.,-1.));
#22 = DIRECTION('',(1.,0.,0.));
#30 = VERTEX_POINT('',#12);
#31 = VERTEX_POINT('',#13);
#40 = AXIS2_PLACEMENT_3D('',#10,#20,#22);
#41 = AXIS2_PLACEMENT_3D('',#11,#20,#22);
#42 = CIRCLE('',#40,2.);
#43 = CIRCLE('',#41,2.);
#44 = VECTOR('',#20,1.);
#45 = LINE('',#12,#44);
#46 = SEAM_CURVE('',#42,(#65,#65),.CURVE_3D.);
#47 = SURFACE_CURVE('',#45,(#65,#63),.CURVE_3D.);
#50 = EDGE_CURVE('',#30,#30,#46,.T.);
#51 = EDGE_CURVE('',#31,#31,#43,.T.);
#52 = EDGE_CURVE('',#30,#31,#47,.T.);
#60 = AXIS2_PLACEMENT_3D('',#10,#21,#22);
#61 = AXIS2_PLACEMENT_3D('',#11,#20,#22);
#62 = PLANE('',#60);
#63 = PLANE('',#61);
#64 = AXIS2_PLACEMENT_3D('',#10,#20,#22);
#65 = CYLINDRICAL_SURFACE('',#64,2.);
#70 = ORIENTED_EDGE('',*,*,#50,.F.);
#71 = ORIENTED_EDGE('',*,*,#51,.T.);
#72 = ORIENTED_EDGE('',*,*,#50,.T.);
#73 = ORIENTED_EDGE('',*,*,#52,.T.);
#74 = ORIENTED_EDGE('',*,*,#51,.F.);
#75 = ORIENTED_EDGE('',*,*,#52,.F.);
#80 = EDGE_LOOP('',(#70));
#81 = EDGE_LOOP('',(#71));
#82 = EDGE_LOOP('',(#72,#73,#74,#75));
#90 = FACE_OUTER_BOUND('',#80,.T.);
#91 = FACE_OUTER_BOUND('',#81,.T.);
#92 = FACE_OUTER_BOUND('',#82,.T.);
#100 = ADVANCED_FACE('',(#90),#62,.T.);
#101 = ADVANCED_FACE('',(#91),#63,.T.);
#102 = ADVANCED_FACE('',(#92),#65,.T.);
#110 = CLOSED_SHELL('',(#100,#101,#102));
#120 = MANIFOLD_SOLID_BREP('cylinder',#110);
ENDSEC;
END-ISO-10303-21;
"#;

#[test]
fn unwraps_surface_curve_edge_geometry() {
    use crate::geometry::cad::brep::curve::Curve;

    let brep = read(CYLINDER_INDIRECT).unwrap();
    assert_eq!(brep.vertices.len(), 2);
    assert_eq!(brep.edges.len(), 3);
    assert_eq!(brep.faces.len(), 3);

    let Curve::Circle(rim) = &brep.edges[0].curve else {
        panic!("rim edge did not resolve through SEAM_CURVE to a circle");
    };
    assert_eq!(rim.radius, 2.0);
    let Curve::Line(_) = &brep.edges[2].curve else {
        panic!("seam edge did not resolve through SURFACE_CURVE to a line");
    };
}

#[test]
fn rejects_missing_solid() {
    let text = "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1 = PLANE('',#2);\n#2 = AXIS2_PLACEMENT_3D('',$,$,$);\nENDSEC;\nEND-ISO-10303-21;\n";
    assert!(
        read(text)
            .err()
            .unwrap()
            .to_string()
            .contains("MANIFOLD_SOLID_BREP")
    );
}
