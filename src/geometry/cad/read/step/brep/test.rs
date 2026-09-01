use super::super::{read, read_all};

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
            Some(6),
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

#[test]
fn read_cylinder_meshes_through_the_analytic_oracle() {
    use crate::{
        geometry::{
            Coordinate,
            cad::sizing::FeatureSizing,
            mesh::{Fitting, Verdict},
            ntree::Balancing,
            solid::{Solid, SolidOracle},
        },
        math::Quantity,
        units::Length,
    };

    // The same STEP body, but forced down the general B-rep path rather than
    // the primitive recogniser: reader-built loops through `BrepOracle`.
    let brep = read(CYLINDER).unwrap();
    let oracle = brep.oracle().unwrap();
    assert!(oracle.signed_distance(&Coordinate::from([0.0, 0.0, 2.5])) > 1.9);
    assert!(oracle.signed_distance(&Coordinate::from([5.0, 0.0, 2.5])) < 0.0);

    let length = |v| Quantity::<Length>::new(v);
    let mesh = brep
        .mesh(
            &FeatureSizing::of(&brep, 32, length(0.2), Some(length(1.0)), Some(0.25)),
            Some(6),
            0.1,
            Balancing::Strong(1),
            Fitting::Soft,
        )
        .unwrap();
    assert!(mesh.minimum_scaled_jacobians()[0].iter().all(|&j| j > 0.0));

    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for coordinate in mesh.coordinates() {
        for k in 0..3 {
            low[k] = low[k].min(coordinate[k].value());
            high[k] = high[k].max(coordinate[k].value());
        }
    }
    assert!(low[0] > -2.5 && high[0] < 2.5);
    assert!(low[2].abs() < 0.6 && (high[2] - 5.0).abs() < 0.6);
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

/// The same sphere, but its numbers are millimetres, declared by an SI_UNIT.
const SPHERE_MILLIMETRES: &str = r#"
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('sphere in mm'),'2;1');
FILE_NAME('sphere.step','2026-08-28T00:00:00',(''),(''),'conspire','conspire','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 }'));
ENDSEC;
DATA;
#10 = CARTESIAN_POINT('',(0.,0.,0.));
#11 = CARTESIAN_POINT('',(0.,0.,-3000.));
#12 = CARTESIAN_POINT('',(0.,0.,3000.));
#20 = DIRECTION('',(0.,0.,1.));
#21 = DIRECTION('',(1.,0.,0.));
#22 = DIRECTION('',(0.,1.,0.));
#30 = VERTEX_POINT('',#11);
#31 = VERTEX_POINT('',#12);
#40 = AXIS2_PLACEMENT_3D('',#10,#22,#21);
#41 = CIRCLE('',#40,3000.);
#50 = EDGE_CURVE('',#30,#31,#41,.T.);
#60 = AXIS2_PLACEMENT_3D('',#10,#20,#21);
#61 = SPHERICAL_SURFACE('',#60,3000.);
#70 = ORIENTED_EDGE('',*,*,#50,.T.);
#71 = ORIENTED_EDGE('',*,*,#50,.F.);
#80 = EDGE_LOOP('',(#70,#71));
#90 = FACE_OUTER_BOUND('',#80,.T.);
#100 = ADVANCED_FACE('',(#90),#61,.T.);
#110 = CLOSED_SHELL('',(#100));
#120 = MANIFOLD_SOLID_BREP('sphere',#110);
#200 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );
ENDSEC;
END-ISO-10303-21;
"#;

/// A file holding two separate solids: a radius-2 sphere at the origin and a
/// radius-3 sphere at `(10, 0, 0)`.
const TWO_SPHERES: &str = r#"
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('two spheres'),'2;1');
FILE_NAME('two.step','2026-08-28T00:00:00',(''),(''),'conspire','conspire','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 }'));
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('',(0.,0.,0.));
#2 = DIRECTION('',(0.,0.,1.));
#3 = DIRECTION('',(1.,0.,0.));
#4 = DIRECTION('',(0.,1.,0.));
#10 = CARTESIAN_POINT('',(0.,0.,-2.));
#11 = CARTESIAN_POINT('',(0.,0.,2.));
#12 = VERTEX_POINT('',#10);
#13 = VERTEX_POINT('',#11);
#14 = AXIS2_PLACEMENT_3D('',#1,#4,#3);
#15 = CIRCLE('',#14,2.);
#16 = EDGE_CURVE('',#12,#13,#15,.T.);
#17 = AXIS2_PLACEMENT_3D('',#1,#2,#3);
#18 = SPHERICAL_SURFACE('',#17,2.);
#19 = ORIENTED_EDGE('',*,*,#16,.T.);
#20 = ORIENTED_EDGE('',*,*,#16,.F.);
#21 = EDGE_LOOP('',(#19,#20));
#22 = FACE_OUTER_BOUND('',#21,.T.);
#23 = ADVANCED_FACE('',(#22),#18,.T.);
#24 = CLOSED_SHELL('',(#23));
#25 = MANIFOLD_SOLID_BREP('A',#24);
#30 = CARTESIAN_POINT('',(10.,0.,0.));
#31 = CARTESIAN_POINT('',(10.,0.,-3.));
#32 = CARTESIAN_POINT('',(10.,0.,3.));
#33 = VERTEX_POINT('',#31);
#34 = VERTEX_POINT('',#32);
#35 = AXIS2_PLACEMENT_3D('',#30,#4,#3);
#36 = CIRCLE('',#35,3.);
#37 = EDGE_CURVE('',#33,#34,#36,.T.);
#38 = AXIS2_PLACEMENT_3D('',#30,#2,#3);
#39 = SPHERICAL_SURFACE('',#38,3.);
#40 = ORIENTED_EDGE('',*,*,#37,.T.);
#41 = ORIENTED_EDGE('',*,*,#37,.F.);
#42 = EDGE_LOOP('',(#40,#41));
#43 = FACE_OUTER_BOUND('',#42,.T.);
#44 = ADVANCED_FACE('',(#43),#39,.T.);
#45 = CLOSED_SHELL('',(#44));
#46 = MANIFOLD_SOLID_BREP('B',#45);
ENDSEC;
END-ISO-10303-21;
"#;

#[test]
#[ignore = "probes local .stp files under STEP_PROBE_DIR, not checked-in fixtures"]
fn probe_step_files() {
    let dir = std::env::var("STEP_PROBE_DIR")
        .unwrap_or_else(|_| format!("{}/../boxy", env!("CARGO_MANIFEST_DIR")));
    let mut files = Vec::new();
    fn walk(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, files);
            } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("stp")) {
                files.push(path);
            }
        }
    }
    walk(std::path::Path::new(&dir), &mut files);
    files.sort();
    if files.is_empty() {
        return;
    }
    let (mut ok, mut fail) = (0, 0);
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy();
        match std::fs::read_to_string(path).map_err(|e| e.to_string()).and_then(|t| read_all(&t).map_err(|e| e.to_string())) {
            Ok(breps) => {
                ok += 1;
                let faces: usize = breps.iter().map(|brep| brep.faces.len()).sum();
                let primitives = breps.iter().filter(|brep| brep.primitive().is_some()).count();
                let assembly = match crate::geometry::cad::assemble::assemble(&breps) {
                    Ok(bodies) => format!("{} bodies", bodies.len()),
                    Err(error) => format!("no ({error})"),
                };
                eprintln!(
                    "ok   {name}: {} solids, {faces} faces, {primitives} primitive, assemble={assembly}",
                    breps.len(),
                );
            }
            Err(error) => {
                fail += 1;
                eprintln!("FAIL {name}: {error}");
            }
        }
    }
    eprintln!("\n{ok} ok, {fail} failed of {}", files.len());
}

/// Trims (or, with `fit`, fully meshes) one solid — a [`Brep`] or a body from
/// [`assemble`](crate::geometry::cad::assemble::assemble) — reporting its
/// element count and, when fitting, its worst scaled Jacobian.
fn mesh_solid(
    solid: &impl crate::geometry::solid::Solid,
    sizing: &impl crate::geometry::solid::Sizing,
    levels: Option<u32>,
    fit: bool,
) -> Result<(usize, Option<f64>), String> {
    use crate::geometry::{
        mesh::{Fitting, Verdict},
        ntree::Balancing,
    };
    if fit {
        let mesh = solid
            .mesh(sizing, levels, 0.1, Balancing::Strong(1), Fitting::Soft)
            .map_err(|e| e.to_string())?;
        let worst = mesh.minimum_scaled_jacobians()[0]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        Ok((mesh.number_of_elements(), Some(worst)))
    } else {
        let (mesh, _) = solid
            .trim(sizing, levels, 0.1, Balancing::Strong(1))
            .map_err(|e| e.to_string())?;
        Ok((mesh.number_of_elements(), None))
    }
}

/// Walks `STEP_MESH_DIR` (default: the checked-in `boxy` dir) and runs the
/// full octree -> dual -> trim -> fit pipeline on every `.stp`, reporting
/// element count and worst scaled Jacobian, or where it fell over. Sizing:
/// `FeatureSizing` with `STEP_MESH_CELL`/`_MIN`/`_SEGMENTS`/`_GRADATION`/
/// `_PROXIMITY` env vars, one per part (no per-file tuning).
#[test]
#[ignore = "meshes every .stp under STEP_MESH_DIR"]
fn probe_mesh_step_dir() {
    use crate::{
        geometry::{
            cad::{assemble::assemble, sizing::FeatureSizing},
            solid::Uniform,
        },
        math::Quantity,
        units::Length,
    };

    let dir = std::env::var("STEP_MESH_DIR")
        .unwrap_or_else(|_| format!("{}/../boxy", env!("CARGO_MANIFEST_DIR")));
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("stp"))
            || path.extension().is_some_and(|e| e.eq_ignore_ascii_case("step"))
        {
            files.push(path);
        }
    }
    files.sort();
    if files.is_empty() {
        return;
    }

    let env_f64 = |key, default: f64| -> f64 {
        std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    };
    let cell = env_f64("STEP_MESH_CELL", 3.0e-3);
    // STEP_MESH_CELL=none => no ceiling: cells grow to the octree root away
    // from the part.
    let maximum = (std::env::var("STEP_MESH_CELL").as_deref() != Ok("none"))
        .then(|| Quantity::<Length>::new(cell));
    let minimum = env_f64("STEP_MESH_MIN", 4.0e-4);
    let segments = env_f64("STEP_MESH_SEGMENTS", 32.0) as usize;
    let gradation = match std::env::var("STEP_MESH_GRADATION").as_deref() {
        Ok("none") => None,
        Ok(v) => Some(v.parse().unwrap()),
        Err(_) => Some(0.2),
    };
    let proximity: Option<usize> = std::env::var("STEP_MESH_PROXIMITY")
        .ok()
        .and_then(|v| v.parse().ok());
    let curvature: Option<usize> = std::env::var("STEP_MESH_CURVATURE")
        .ok()
        .and_then(|v| v.parse().ok());
    // Off by default: dual + trim only, the geometry-pipeline signal.
    let fit = std::env::var("STEP_MESH_FIT").is_ok();

    let (mut meshed, mut failed) = (0, 0);
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let started = std::time::Instant::now();
        let outcome = (|| -> Result<(usize, Option<f64>), String> {
            let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
            let breps = read_all(&text).map_err(|e| e.to_string())?;
            let (mut elements, mut worst) = (0usize, f64::INFINITY);
            // A recognised assembly (every solid a primitive, interior solids
            // carved as voids) meshes body by body with a uniform field;
            // anything else meshes solid by solid with feature sizing. Element
            // counts and worst SJ are summed over the whole set.
            if let Ok(bodies) = assemble(&breps) {
                let sizing = Uniform(maximum.unwrap_or_else(|| Quantity::<Length>::new(cell)));
                for body in &bodies {
                    let (n, w) = mesh_solid(body, &sizing, None, fit)?;
                    elements += n;
                    worst = worst.min(w.unwrap_or(f64::INFINITY));
                }
            } else {
                for brep in &breps {
                    // Fail fast on an unmeshable face before paying for the octree.
                    brep.oracle().map_err(|e| e.to_string())?;
                    let mut sizing = FeatureSizing::of(
                        brep,
                        segments,
                        Quantity::<Length>::new(minimum),
                        maximum,
                        gradation,
                    );
                    if let Some(cells) = proximity {
                        sizing = sizing.with_proximity(brep, cells).map_err(|e| e.to_string())?;
                    }
                    if let Some(sections) = curvature {
                        sizing = sizing.with_curvature(brep, sections).map_err(|e| e.to_string())?;
                    }
                    let (n, w) = mesh_solid(brep, &sizing, None, fit)?;
                    elements += n;
                    worst = worst.min(w.unwrap_or(f64::INFINITY));
                }
            }
            Ok((elements, fit.then_some(worst)))
        })();
        let secs = started.elapsed().as_secs_f64();
        match outcome {
            Ok((elements, worst)) => {
                meshed += 1;
                match worst {
                    Some(worst) => eprintln!(
                        "ok   {name}: {elements} elements, worst SJ {worst:.4} ({secs:.0}s)"
                    ),
                    None => eprintln!("ok   {name}: {elements} trimmed hexes ({secs:.0}s)"),
                }
            }
            Err(error) => {
                failed += 1;
                eprintln!("FAIL {name}: {error} ({secs:.0}s)");
            }
        }
    }
    eprintln!("\n{meshed} ok, {failed} failed of {}", files.len());
}

/// Runs the octree -> dual -> trim (-> fit) pipeline on `brep` with `sizing`,
/// writing each stage to `{out}_{dual,trimmed,fitted}.vtu` for ParaView. The
/// trimmed dump is the one to look at before paying for the fit.
fn probe_mesh(
    brep: &crate::geometry::cad::brep::Brep,
    sizing: &impl crate::geometry::solid::Sizing,
    levels: Option<u32>,
    out: &str,
    fit: bool,
) {
    use crate::{
        geometry::{
            mesh::{Class, Fitting, Output, Verdict, Vtk},
            ntree::Balancing,
            solid::Solid,
        },
        io::{Write, write::Compression},
    };

    let dump = |mesh: &crate::geometry::mesh::Mesh<3>, path: &str| {
        mesh.write(Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(path))))
            .unwrap();
        eprintln!("wrote {path}");
    };
    let started = std::time::Instant::now();

    // STEP_MESH_OCTREE_ONLY: dump the refined sizing octree and stop, so the
    // sizing field can be inspected without the classify/dual grind.
    if std::env::var("STEP_MESH_OCTREE_ONLY").is_ok() {
        let octree = brep
            .sizing_octree(sizing, levels, 0.1)
            .expect("sizing_octree failed");
        eprintln!(
            "octree: {} leaves ({:.1}s)",
            octree.number_of_elements(),
            started.elapsed().as_secs_f64(),
        );
        // Mirror-pair leaf census: count leaves and sum 1/size^3 in a box at
        // +STEP_MIRROR and its reflection across the plane x = STEP_MIRROR_AT,
        // so an asymmetry in the raw octree (before classify/dual/trim) shows
        // up as a count mismatch here.
        if let Ok(m) = std::env::var("STEP_MIRROR") {
            let m: f64 = m.parse().unwrap();
            let at: f64 = std::env::var("STEP_MIRROR_AT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
            let rad: f64 = std::env::var("STEP_MIRROR_RAD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10.0e-3);
            let zc: f64 = std::env::var("STEP_MIRROR_Z")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(40.0e-3);
            let coords = octree.coordinates();
            let (mut np, mut nm) = (0usize, 0usize);
            let (mut lp, mut lm) = ([0.0f64; 12], [0.0f64; 12]);
            for block in octree.iter() {
                for element in block.iter() {
                    let nodes = block.element_nodes(element);
                    let c: [f64; 3] = std::array::from_fn(|k| {
                        nodes.iter().map(|&n| coords[n][k].value()).sum::<f64>() / nodes.len() as f64
                    });
                    let h = (coords[nodes[0]][0].value() - c[0]).abs() * 2.0;
                    let lvl = (h.log2().round() as i64).rem_euclid(12) as usize;
                    let near = |x: f64| {
                        (c[1] - 0.0).abs() < rad
                            && (c[2] - zc).abs() < rad
                            && (c[0] - x).abs() < rad
                    };
                    if near(at + m) {
                        np += 1;
                        lp[lvl] += 1.0;
                    }
                    if near(at - m) {
                        nm += 1;
                        lm[lvl] += 1.0;
                    }
                }
            }
            eprintln!("mirror +{m}: {np} leaves, by size-bucket {lp:?}");
            eprintln!("mirror -{m}: {nm} leaves, by size-bucket {lm:?}");
        }
        dump(&octree, &format!("{out}_octree.vtu"));
        return;
    }

    let (dual, classes) = brep
        .dual_background(sizing, levels, 0.1, Balancing::Strong(1))
        .expect("dual_background failed");
    let count = |class| classes.iter().filter(|&&c| c == class).count();
    eprintln!(
        "dual: {} hexes ({:.1}s); {} inside, {} cut, {} outside",
        dual.number_of_elements(),
        started.elapsed().as_secs_f64(),
        count(Class::Inside),
        count(Class::Cut),
        count(Class::Outside),
    );
    dump(&dual, &format!("{out}_dual.vtu"));

    // Split the classified dual so the Inside-only and Cut-only cells can be
    // eyeballed apart (a phantom column is usually one or the other).
    for (label, want) in [("inside", Class::Inside), ("cut", Class::Cut)] {
        let (mut only, only_classes) = brep
            .dual_background(sizing, levels, 0.1, Balancing::Strong(1))
            .expect("dual_background failed");
        only
            .keep_hexes(|index, _, _| only_classes[index] == want)
            .expect("keep_hexes failed");
        eprintln!("  {label}: {} hexes", only.number_of_elements());
        dump(&only, &format!("{out}_{label}.vtu"));
    }

    let (trimmed, _) = brep
        .trim(sizing, levels, 0.1, Balancing::Strong(1))
        .expect("trim failed");
    // Mirror-pair census of the dual and the trimmed mesh: `Inside`+`Cut`
    // counts in a box at +/-STEP_MIRROR across x=STEP_MIRROR_AT. First stage
    // that mismatches is the one breaking symmetry.
    if let Ok(m) = std::env::var("STEP_MIRROR") {
        let m: f64 = m.parse().unwrap();
        let at: f64 = std::env::var("STEP_MIRROR_AT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let rad: f64 = std::env::var("STEP_MIRROR_RAD").ok().and_then(|v| v.parse().ok()).unwrap_or(10.0e-3);
        let zc: f64 = std::env::var("STEP_MIRROR_Z").ok().and_then(|v| v.parse().ok()).unwrap_or(40.0e-3);
        let census = |mesh: &crate::geometry::mesh::Mesh<3>, tag: &str| {
            let coords = mesh.coordinates();
            let (mut p, mut n) = (0usize, 0usize);
            for block in mesh.iter() {
                for element in block.iter() {
                    let nodes = block.element_nodes(element);
                    let c: [f64; 3] = std::array::from_fn(|k| {
                        nodes.iter().map(|&i| coords[i][k].value()).sum::<f64>() / nodes.len() as f64
                    });
                    if (c[1]).abs() < rad && (c[2] - zc).abs() < rad {
                        if (c[0] - (at + m)).abs() < rad {
                            p += 1;
                        }
                        if (c[0] - (at - m)).abs() < rad {
                            n += 1;
                        }
                    }
                }
            }
            eprintln!("{tag}: +{m} -> {p} hexes, -{m} -> {n} hexes  (diff {})", p as i64 - n as i64);
        };
        census(&dual, "dual  ");
        census(&trimmed, "trimmed");
    }
    eprintln!(
        "trimmed: {} hexes ({:.1}s total)",
        trimmed.number_of_elements(),
        started.elapsed().as_secs_f64(),
    );
    dump(&trimmed, &format!("{out}_trimmed.vtu"));

    if !fit {
        return;
    }
    let mesh = brep
        .mesh(sizing, levels, 0.1, Balancing::Strong(1), Fitting::Soft)
        .expect("mesh failed");
    let worst = mesh.minimum_scaled_jacobians()[0]
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    eprintln!(
        "meshed: {} nodes, {} elements, worst scaled Jacobian {worst}",
        mesh.number_of_nodes(),
        mesh.number_of_elements(),
    );
    dump(&mesh, &format!("{out}_fitted.vtu"));
}

/// Samples `FeatureSizing::at_cell` on a circle of radius `STEP_RING_R` about
/// the line `(t, 0, STEP_RING_Z)`, at every axial `STEP_RING_X` (comma list)
/// and every `STEP_RING_STEP` degrees, printing size vs angle so a curved-face
/// sizing band can be checked for rotational symmetry (and two mirrored bores
/// compared) without opening the mesh.
#[test]
#[ignore = "samples the sizing field around a ring, STEP_MESH_FILE + STEP_RING_*"]
fn probe_sizing_ring() {
    use crate::{
        geometry::{Coordinate, cad::sizing::FeatureSizing},
        math::Quantity,
        units::Length,
    };
    let Ok(path) = std::env::var("STEP_MESH_FILE") else {
        return;
    };
    let env_f64 = |key, default: f64| {
        std::env::var(key).ok().and_then(|v: String| v.parse().ok()).unwrap_or(default)
    };
    let brep = read(&std::fs::read_to_string(&path).unwrap()).expect("read failed");
    let length = |v| Quantity::<Length>::new(v);
    let cell = env_f64("STEP_MESH_CELL", 8.0e-3);
    let sizing = FeatureSizing::of(
        &brep,
        env_f64("STEP_MESH_SEGMENTS", 36.0) as usize,
        length(env_f64("STEP_MESH_MIN", 6.0e-4)),
        Some(length(cell)),
        Some(env_f64("STEP_MESH_GRADATION", 0.15)),
    )
    .with_proximity(&brep, env_f64("STEP_MESH_PROXIMITY", 3.0) as usize)
    .unwrap()
    .with_curvature(&brep, env_f64("STEP_MESH_CURVATURE", 48.0) as usize)
    .unwrap();

    let radius = env_f64("STEP_RING_R", 5.3e-3);
    let ring_z = env_f64("STEP_RING_Z", 40.0e-3);
    let half = env_f64("STEP_RING_HALF", 0.4e-3);
    let step = env_f64("STEP_RING_STEP", 10.0);
    let xs: Vec<f64> = std::env::var("STEP_RING_X")
        .unwrap_or_else(|_| "31e-3,-31e-3".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    for x in xs {
        eprintln!("--- axial x = {x:.5}, radius {radius:.5} ---");
        let mut deg = 0.0_f64;
        while deg < 360.0 {
            let t = deg.to_radians();
            let p = Coordinate::from([x, radius * t.cos(), ring_z + radius * t.sin()]);
            eprintln!("  {deg:6.1} deg  size = {:.6}", sizing.at_cell(&p, half).value());
            deg += step;
        }
    }
}

#[test]
#[ignore = "meshes a local .stp given by STEP_MESH_FILE"]
fn probe_mesh_real_file() {
    use crate::{
        geometry::{cad::sizing::FeatureSizing, solid::Uniform},
        math::Quantity,
        units::Length,
    };

    let Ok(path) = std::env::var("STEP_MESH_FILE") else {
        return;
    };
    let text = std::fs::read_to_string(&path).unwrap();
    let breps = read_all(&text).expect("read failed");
    eprintln!(
        "read {} solid(s), {} faces total",
        breps.len(),
        breps.iter().map(|brep| brep.faces.len()).sum::<usize>(),
    );

    let env_f64 = |key, default: f64| -> f64 {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let length = |v| Quantity::<Length>::new(v);
    let cell = env_f64("STEP_MESH_CELL", 6.0e-3);
    let out = std::env::var("STEP_MESH_OUT").unwrap_or_else(|_| "target/step_mesh".into());
    let fit = std::env::var("STEP_MESH_FIT").is_ok();
    // Unset STEP_MESH_LEVELS => None => refine as far as the sizing field wants.
    let levels = std::env::var("STEP_MESH_LEVELS")
        .ok()
        .and_then(|value| value.parse().ok());

    // One dump set per solid; suffix the prefix when the file holds more than
    // one so the VTUs do not collide.
    for (index, brep) in breps.iter().enumerate() {
        let out = if breps.len() == 1 {
            out.clone()
        } else {
            format!("{out}_solid{index}")
        };
        eprintln!("--- solid {index}: {} faces -> {out} ---", brep.faces.len());

        // STEP_MESH_SIZING=uniform for a flat field; feature (default) drives
        // refinement from the B-rep's sharp edges.
        if std::env::var("STEP_MESH_SIZING").as_deref() == Ok("uniform") {
            probe_mesh(brep, &Uniform(length(cell)), levels, &out, fit);
            continue;
        }
        // STEP_MESH_GRADATION="none" => grade as fast as it likes (one fine
        // layer per feature); a number => that bounded rate; unset => 0.2.
        let gradation = match std::env::var("STEP_MESH_GRADATION").as_deref() {
            Ok("none") => None,
            Ok(value) => Some(value.parse().expect("STEP_MESH_GRADATION")),
            Err(_) => Some(0.2),
        };
        // STEP_MESH_CELL=none => no ceiling: cells grow to the octree root
        // away from the part.
        let maximum = (std::env::var("STEP_MESH_CELL").as_deref() != Ok("none"))
            .then(|| length(cell));
        let mut sizing = FeatureSizing::of(
            brep,
            env_f64("STEP_MESH_SEGMENTS", 24.0) as usize,
            length(env_f64("STEP_MESH_MIN", cell / 8.0)),
            maximum,
            gradation,
        );
        // STEP_MESH_PROXIMITY=N adds the local-feature-size term (N cells
        // across a thin wall or narrow cavity).
        if let Ok(n) = std::env::var("STEP_MESH_PROXIMITY") {
            let t = std::time::Instant::now();
            sizing = sizing
                .with_proximity(brep, n.parse().expect("STEP_MESH_PROXIMITY"))
                .expect("with_proximity");
            eprintln!("with_proximity built in {:.1}s", t.elapsed().as_secs_f64());
        }
        // STEP_MESH_CURVATURE=N resolves every curved face at N cells around a
        // full circle of its local curvature radius.
        if let Ok(n) = std::env::var("STEP_MESH_CURVATURE") {
            let t = std::time::Instant::now();
            sizing = sizing
                .with_curvature(brep, n.parse().expect("STEP_MESH_CURVATURE"))
                .expect("with_curvature");
            eprintln!("with_curvature built in {:.1}s", t.elapsed().as_secs_f64());
        }
        probe_mesh(brep, &sizing, levels, &out, fit);
    }
}

/// Prints the `signed_distance` sign along evenly spaced lines through the
/// bbox on each axis: `#` inside (sd > 0), `.` outside. A phantom string shows
/// up as `#` runs where the geometry is a void.
#[test]
#[ignore = "prints STEP_MESH_FILE's signed-distance sign along scan lines"]
fn probe_signed_distance_sign() {
    use crate::geometry::{Coordinate, solid::SolidOracle};

    let Ok(path) = std::env::var("STEP_MESH_FILE") else {
        return;
    };
    let brep = read(&std::fs::read_to_string(&path).unwrap()).expect("read failed");
    let oracle = brep.oracle().expect("oracle failed");
    let (low, high) = oracle.bounds();
    let span: [f64; 3] = std::array::from_fn(|k| high[k].value() - low[k].value());
    let base: [f64; 3] = std::array::from_fn(|k| low[k].value());

    eprintln!(
        "bbox low=[{:.3},{:.3},{:.3}] high=[{:.3},{:.3},{:.3}]  {} faces",
        low[0].value(), low[1].value(), low[2].value(),
        high[0].value(), high[1].value(), high[2].value(),
        brep.faces.len(),
    );
    for (fi, face) in brep.faces.iter().enumerate() {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for lp in &face.bounds {
            for he in &lp.half_edges {
                for vi in brep.edges[he.edge].vertices {
                    let v = &brep.vertices[vi];
                    for k in 0..3 {
                        lo[k] = lo[k].min(v[k].value());
                        hi[k] = hi[k].max(v[k].value());
                    }
                }
            }
        }
        let kind = match &face.surface {
            crate::geometry::cad::brep::surface::Surface::Plane(p) => format!(
                "plane n=[{:.2},{:.2},{:.2}]",
                p.normal[0].value(), p.normal[1].value(), p.normal[2].value()
            ),
            crate::geometry::cad::brep::surface::Surface::Cylinder(_) => "cylinder".into(),
            crate::geometry::cad::brep::surface::Surface::Cone(_) => "cone".into(),
            crate::geometry::cad::brep::surface::Surface::Sphere(_) => "sphere".into(),
            crate::geometry::cad::brep::surface::Surface::Torus(_) => "torus".into(),
            _ => "bspline".into(),
        };
        eprintln!(
            "  f{fi:<3} fwd={} {:<28} bbox=[{:.3},{:.3},{:.3}]..[{:.3},{:.3},{:.3}]",
            face.forward as u8, kind,
            lo[0], lo[1], lo[2], hi[0], hi[1], hi[2],
        );
    }
    let samples = 100usize;
    let lines = 9usize;

    for axis in 0..3 {
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        eprintln!("--- scan along axis {axis} ---");
        for a in 1..lines {
            for b in 1..lines {
                let mut row = String::new();
                for s in 0..samples {
                    let mut p = [0.0; 3];
                    p[axis] = base[axis] + span[axis] * (s as f64 + 0.5) / samples as f64;
                    p[u] = base[u] + span[u] * a as f64 / lines as f64;
                    p[v] = base[v] + span[v] * b as f64 / lines as f64;
                    let q = Coordinate::from(p);
                    row.push(if oracle.signed_distance(&q) > 0.0 {
                        let ld = oracle.local_diameter(&q);
                        match ld {
                            _ if ld < 0.005 => '1',
                            _ if ld < 0.010 => '2',
                            _ if ld < 0.020 => '3',
                            _ if ld < 0.040 => '4',
                            _ if ld < 0.080 => '5',
                            _ => '#',
                        }
                    } else {
                        '.'
                    });
                }
                if row.contains('#') {
                    eprintln!("u{a} v{b} {row}");
                }
            }
        }
    }

    let centre = crate::geometry::Coordinate::from(std::array::from_fn::<f64, 3, _>(|k| {
        0.5 * (low[k].value() + high[k].value())
    }));
    eprintln!("centre: signed_distance = {:.5}", oracle.signed_distance(&centre));
    if let Ok(spec) = std::env::var("STEP_PROBE_POINTS") {
        for chunk in spec.split(';') {
            let c: Vec<f64> = chunk.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            if c.len() != 3 {
                continue;
            }
            let p = crate::geometry::Coordinate::from([c[0], c[1], c[2]]);
            let (kind, d, pt, n) = oracle.patch_report(&p).into_iter().next().unwrap();
            eprintln!(
                "probe {c:?}: sd={:.5} local_diameter={:.5}  nearest {kind} d={d:.5} at [{:.4},{:.4},{:.4}] n=[{:.2},{:.2},{:.2}]",
                oracle.signed_distance(&p),
                oracle.local_diameter(&p),
                pt[0], pt[1], pt[2], n[0], n[1], n[2],
            );
        }
    }

    // Is it a clean global flip, or per-region inconsistency? Tally the sign at
    // many points deep inside (near the centre) vs far outside (past a face).
    let mut inside_pos = 0;
    let mut inside_neg = 0;
    let mut outside_pos = 0;
    let mut outside_neg = 0;
    for i in 0..7 {
        for j in 0..7 {
            for k in 0..7 {
                let f = |t: usize| (t as f64 + 0.5) / 7.0;
                let deep = crate::geometry::Coordinate::from([
                    centre[0].value() + span[0] * 0.20 * (f(i) - 0.5),
                    centre[1].value() + span[1] * 0.20 * (f(j) - 0.5),
                    centre[2].value() + span[2] * 0.20 * (f(k) - 0.5),
                ]);
                if oracle.signed_distance(&deep) > 0.0 { inside_pos += 1 } else { inside_neg += 1 }
                let far = crate::geometry::Coordinate::from([
                    low[0].value() - span[0] * (0.3 + f(i)),
                    low[1].value() + span[1] * f(j),
                    low[2].value() + span[2] * f(k),
                ]);
                if oracle.signed_distance(&far) > 0.0 { outside_pos += 1 } else { outside_neg += 1 }
            }
        }
    }
    eprintln!("deep-inside points:  {inside_pos} positive, {inside_neg} negative (want all positive)");
    eprintln!("far-outside points:  {outside_pos} positive, {outside_neg} negative (want all negative)");
}

/// Falsifies a wrong ray-parity sign without a reference classifier: a lost or
/// spurious crossing flips a whole shadow region, whose boundary is then a sign
/// change with no surface within the sampling step. Reports every such pair.
#[test]
#[ignore = "sweeps STEP_MESH_FILE for sign flips away from any surface"]
fn probe_sign_consistency() {
    use crate::geometry::{Coordinate, solid::SolidOracle};
    let path = std::env::var("STEP_MESH_FILE").unwrap();
    let brep = read(&std::fs::read_to_string(&path).unwrap()).expect("read failed");
    let oracle = brep.oracle().expect("oracle failed");
    let (low, high) = oracle.bounds();
    let span: [f64; 3] = std::array::from_fn(|k| high[k].value() - low[k].value());
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut rand = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let step = span[0] * 1.0e-4;
    let mut bad = 0;
    for _ in 0..400000 {
        let p: [f64; 3] =
            std::array::from_fn(|k| low[k].value() + span[k] * rand());
        let mut q = p;
        let axis = (rand() * 3.0) as usize % 3;
        q[axis] += step;
        let (a, b) = (
            oracle.signed_distance(&Coordinate::from(p)),
            oracle.signed_distance(&Coordinate::from(q)),
        );
        if (a > 0.0) != (b > 0.0) && a.abs().min(b.abs()) > step {
            bad += 1;
            if bad <= 20 {
                eprintln!(
                    "flip without a surface: [{:.5},{:.5},{:.5}] sd={a:.6} -> axis{axis} sd={b:.6}",
                    p[0], p[1], p[2]
                );
            }
        }
    }
    eprintln!("{bad} inconsistent pairs of 400000");
}

/// Every face crossing along each of `encloses`'s three ray directions, so a
/// disputed sign can be read off the crossing count face by face.
#[test]
#[ignore = "dumps per-face ray hits at STEP_PROBE_POINTS"]
fn probe_ray_hits() {
    use crate::geometry::{Coordinate, solid::SolidOracle};
    let path = std::env::var("STEP_MESH_FILE").unwrap();
    let brep = read(&std::fs::read_to_string(&path).unwrap()).expect("read failed");
    let oracle = brep.oracle().expect("oracle failed");
    let dirs = [
        [0.862_667, 0.411_988, 0.291_536],
        [0.301_511, 0.904_534, 0.301_511],
        [0.334_412, 0.243_975, 0.910_367],
    ];
    for chunk in std::env::var("STEP_PROBE_POINTS").unwrap().split(';') {
        let c: Vec<f64> = chunk.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        if c.len() != 3 {
            continue;
        }
        let p = Coordinate::from([c[0], c[1], c[2]]);
        eprintln!("=== {c:?} sd={:.6}", oracle.signed_distance(&p));
        for (di, d) in dirs.into_iter().enumerate() {
            let rows = oracle.ray_report(&p, d);
            eprintln!("  dir{di} {} hits", rows.len());
            for (index, kind, t) in rows {
                let hit: [f64; 3] = std::array::from_fn(|k| c[k] + t * d[k]);
                eprintln!(
                    "    f{index:<3} {kind:<6} t={t:.6} at [{:.4},{:.4},{:.4}]",
                    hit[0], hit[1], hit[2]
                );
            }
        }
    }
}

#[test]
fn reads_every_solid_in_the_file() {
    use crate::geometry::{cad::brep::surface::Surface, csg::Primitive};

    let breps = read_all(TWO_SPHERES).unwrap();
    assert_eq!(breps.len(), 2);

    let radii: Vec<f64> = breps
        .iter()
        .map(|brep| {
            assert!(
                matches!(brep.primitive(), Some(Primitive::Sphere(_))),
                "solid not recognised as a sphere"
            );
            let Surface::Sphere(sphere) = &brep.faces[0].surface else {
                unreachable!()
            };
            sphere.radius
        })
        .collect();
    assert_eq!(radii, vec![2.0, 3.0]);

    // The single-solid `read` refuses a multi-solid file.
    assert!(read(TWO_SPHERES).is_err());
}

/// A solid whose one non-trivial edge is a cubic B-spline.
const BSPLINE_EDGE: &str = r#"
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('bspline edge'),'2;1');
FILE_NAME('b.step','2026-08-29T00:00:00',(''),(''),'conspire','conspire','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#10 = CARTESIAN_POINT('',(0.,0.,0.));
#11 = CARTESIAN_POINT('',(0.,0.,-3.));
#12 = CARTESIAN_POINT('',(0.,0.,3.));
#13 = CARTESIAN_POINT('',(3.,0.,-1.));
#14 = CARTESIAN_POINT('',(3.,0.,1.));
#20 = DIRECTION('',(0.,0.,1.));
#21 = DIRECTION('',(1.,0.,0.));
#30 = VERTEX_POINT('',#11);
#31 = VERTEX_POINT('',#12);
#41 = B_SPLINE_CURVE_WITH_KNOTS('',3,(#11,#13,#14,#12),.UNSPECIFIED.,.F.,.F.,(4,4),(0.,1.),.UNSPECIFIED.);
#50 = EDGE_CURVE('',#30,#31,#41,.T.);
#60 = AXIS2_PLACEMENT_3D('',#10,#20,#21);
#61 = SPHERICAL_SURFACE('',#60,3.);
#70 = ORIENTED_EDGE('',*,*,#50,.T.);
#71 = ORIENTED_EDGE('',*,*,#50,.F.);
#80 = EDGE_LOOP('',(#70,#71));
#90 = FACE_OUTER_BOUND('',#80,.T.);
#100 = ADVANCED_FACE('',(#90),#61,.T.);
#110 = CLOSED_SHELL('',(#100));
#120 = MANIFOLD_SOLID_BREP('b',#110);
ENDSEC;
END-ISO-10303-21;
"#;

#[test]
fn reads_a_bspline_curve_raw() {
    use crate::geometry::cad::brep::curve::Curve;

    let brep = read(BSPLINE_EDGE).unwrap();
    let Curve::BSpline(spline) = &brep.edges[0].curve else {
        panic!("edge is not a B-spline");
    };
    assert_eq!(spline.degree, 3);
    assert_eq!(spline.control_points.len(), 4);
    assert_eq!(spline.multiplicities, vec![4, 4]);
    assert_eq!(spline.knots, vec![0.0, 1.0]);
    assert!(spline.weights.is_none());
    // Control points are read (and would be unit-scaled).
    assert_eq!(spline.control_points[1][0].value(), 3.0);
}

/// A knotless (implied-knot) B-spline surface: degree 1 in both directions,
/// a 4x2 control grid. `u` has 3 segments (exercises the real quasi-uniform
/// ladder); `v` has 1 (falls to the clamped-ends default, same as a Bezier).
const QUASI_UNIFORM_SURFACE_FACE: &str = r#"
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('quasi-uniform surface'),'2;1');
FILE_NAME('q.step','2026-09-01T00:00:00',(''),(''),'conspire','conspire','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#10 = CARTESIAN_POINT('',(0.,0.,-1.));
#11 = CARTESIAN_POINT('',(0.,0.,1.));
#20 = VERTEX_POINT('',#10);
#21 = VERTEX_POINT('',#11);
#30 = DIRECTION('',(0.,0.,1.));
#31 = VECTOR('',#30,1.);
#40 = LINE('',#10,#31);
#50 = EDGE_CURVE('',#20,#21,#40,.T.);
#200 = CARTESIAN_POINT('',(-1.,-1.,0.));
#201 = CARTESIAN_POINT('',(-1.,1.,0.));
#202 = CARTESIAN_POINT('',(0.,-1.,0.5));
#203 = CARTESIAN_POINT('',(0.,1.,0.5));
#204 = CARTESIAN_POINT('',(1.,-1.,1.));
#205 = CARTESIAN_POINT('',(1.,1.,1.));
#206 = CARTESIAN_POINT('',(2.,-1.,1.5));
#207 = CARTESIAN_POINT('',(2.,1.,1.5));
#210 = QUASI_UNIFORM_SURFACE('',1,1,((#200,#201),(#202,#203),(#204,#205),(#206,#207)),
   .UNSPECIFIED.,.F.,.F.,.U.);
#70 = ORIENTED_EDGE('',*,*,#50,.T.);
#71 = ORIENTED_EDGE('',*,*,#50,.F.);
#80 = EDGE_LOOP('',(#70,#71));
#90 = FACE_OUTER_BOUND('',#80,.T.);
#100 = ADVANCED_FACE('',(#90),#210,.T.);
#110 = CLOSED_SHELL('',(#100));
#120 = MANIFOLD_SOLID_BREP('q',#110);
ENDSEC;
END-ISO-10303-21;
"#;

#[test]
fn reads_a_quasi_uniform_surface_with_implied_knots() {
    use crate::geometry::cad::brep::surface::Surface;

    let brep = read(QUASI_UNIFORM_SURFACE_FACE).unwrap();
    let Surface::BSpline(surface) = &brep.faces[0].surface else {
        panic!("face is not a B-spline surface");
    };
    assert_eq!(surface.u_degree, 1);
    assert_eq!(surface.v_degree, 1);
    assert_eq!(surface.control_points.len(), 4);
    assert_eq!(surface.control_points[0].len(), 2);
    // u: degree 1, 4 control points -> 3 segments, so the real quasi-uniform
    // ladder (clamped ends, single interior knots) applies.
    assert_eq!(surface.u_knots, vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(surface.u_multiplicities, vec![2, 1, 1, 2]);
    // v: degree 1, 2 control points -> 1 segment, too few to ladder, so it
    // falls to the same clamped-ends default a Bezier gets.
    assert_eq!(surface.v_knots, vec![0.0, 1.0]);
    assert_eq!(surface.v_multiplicities, vec![2, 2]);
    assert!(surface.weights.is_none());
}

const SURFACE_OF_REVOLUTION_FACE: &str = r#"
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('surface of revolution'),'2;1');
FILE_NAME('r.step','2026-09-01T00:00:00',(''),(''),'conspire','conspire','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#10 = CARTESIAN_POINT('',(1.,0.,0.));
#11 = CARTESIAN_POINT('',(1.,0.,2.));
#20 = VERTEX_POINT('',#10);
#21 = VERTEX_POINT('',#11);
#30 = DIRECTION('',(0.,0.,1.));
#31 = VECTOR('',#30,1.);
#40 = LINE('',#10,#31);
#50 = EDGE_CURVE('',#20,#21,#40,.T.);
#60 = CARTESIAN_POINT('',(0.,0.,0.));
#61 = DIRECTION('',(0.,0.,1.));
#62 = AXIS1_PLACEMENT('',#60,#61);
#63 = SURFACE_OF_REVOLUTION('',#40,#62);
#80 = ORIENTED_EDGE('',*,*,#50,.T.);
#81 = ORIENTED_EDGE('',*,*,#50,.F.);
#90 = EDGE_LOOP('',(#80,#81));
#95 = FACE_OUTER_BOUND('',#90,.T.);
#100 = ADVANCED_FACE('',(#95),#63,.T.);
#110 = CLOSED_SHELL('',(#100));
#120 = MANIFOLD_SOLID_BREP('r',#110);
ENDSEC;
END-ISO-10303-21;
"#;

#[test]
fn reads_a_surface_of_revolution() {
    use crate::geometry::{
        Coordinate, Direction,
        cad::brep::{curve::Curve, surface::Surface},
    };

    let brep = read(SURFACE_OF_REVOLUTION_FACE).unwrap();
    let Surface::Revolution(revolution) = &brep.faces[0].surface else {
        panic!("face is not a surface of revolution");
    };
    assert!(matches!(revolution.curve, Curve::Line(_)));
    assert_eq!(revolution.origin, Coordinate::from([0.0, 0.0, 0.0]));
    assert_eq!(revolution.axis, Direction::from([0.0, 0.0, 1.0]));
}

#[test]
fn scales_coordinates_from_the_declared_length_unit() {
    use crate::geometry::cad::brep::surface::Surface;

    let brep = read(SPHERE_MILLIMETRES).unwrap();
    let Surface::Sphere(sphere) = &brep.faces[0].surface else {
        panic!("face is not spherical");
    };
    // 3000 mm read back as 3 m.
    assert!((sphere.radius - 3.0).abs() < 1e-12);
    assert!((brep.vertices[0][2].value() + 3.0).abs() < 1e-12);
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
fn unwraps_trimmed_and_implied_knot_curves() {
    use crate::geometry::cad::brep::curve::Curve;

    let text = CYLINDER_INDIRECT.replace(
        "#47 = SURFACE_CURVE('',#45,(#65,#63),.CURVE_3D.);",
        "#47 = TRIMMED_CURVE('',#48,(PARAMETER_VALUE(0.)),(PARAMETER_VALUE(1.)),.T.,.UNSPECIFIED.);\n\
         #48 = QUASI_UNIFORM_CURVE('',1,(#12,#13),.UNSPECIFIED.,.F.,.U.);",
    );
    let brep = read(&text).unwrap();
    let Curve::BSpline(seam) = &brep.edges[2].curve else {
        panic!("seam edge did not resolve through TRIMMED_CURVE to a B-spline");
    };
    assert_eq!(seam.degree, 1);
    assert_eq!(seam.knots, vec![0.0, 1.0]);
    assert_eq!(seam.multiplicities, vec![2, 2]);
    let middle = seam.point(0.5);
    assert!((middle[0].value() - 2.0).abs() < 1.0e-12);
    assert!((middle[2].value() - 2.5).abs() < 1.0e-12);
    assert!(brep.oracle().is_ok());
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
