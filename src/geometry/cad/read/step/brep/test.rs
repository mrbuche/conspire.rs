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
            } else if path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("stp"))
            {
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
        match std::fs::read_to_string(path)
            .map_err(|e| e.to_string())
            .and_then(|t| read_all(&t).map_err(|e| e.to_string()))
        {
            Ok(breps) => {
                ok += 1;
                let faces: usize = breps.iter().map(|brep| brep.faces.len()).sum();
                let primitives = breps
                    .iter()
                    .filter(|brep| brep.primitive().is_some())
                    .count();
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
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("stp"))
            || path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("step"))
        {
            files.push(path);
        }
    }
    files.sort();
    if files.is_empty() {
        return;
    }

    let env_f64 = |key, default: f64| -> f64 {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
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
                        sizing = sizing
                            .with_proximity(brep, cells)
                            .map_err(|e| e.to_string())?;
                    }
                    if let Some(sections) = curvature {
                        sizing = sizing
                            .with_curvature(brep, sections)
                            .map_err(|e| e.to_string())?;
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
                        nodes.iter().map(|&n| coords[n][k].value()).sum::<f64>()
                            / nodes.len() as f64
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
        only.keep_hexes(|index, _, _| only_classes[index] == want)
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
        let census = |mesh: &crate::geometry::mesh::Mesh<3>, tag: &str| {
            let coords = mesh.coordinates();
            let (mut p, mut n) = (0usize, 0usize);
            for block in mesh.iter() {
                for element in block.iter() {
                    let nodes = block.element_nodes(element);
                    let c: [f64; 3] = std::array::from_fn(|k| {
                        nodes.iter().map(|&i| coords[i][k].value()).sum::<f64>()
                            / nodes.len() as f64
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
            eprintln!(
                "{tag}: +{m} -> {p} hexes, -{m} -> {n} hexes  (diff {})",
                p as i64 - n as i64
            );
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
        std::env::var(key)
            .ok()
            .and_then(|v: String| v.parse().ok())
            .unwrap_or(default)
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
            eprintln!(
                "  {deg:6.1} deg  size = {:.6}",
                sizing.at_cell(&p, half).value()
            );
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
        let maximum =
            (std::env::var("STEP_MESH_CELL").as_deref() != Ok("none")).then(|| length(cell));
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
        low[0].value(),
        low[1].value(),
        low[2].value(),
        high[0].value(),
        high[1].value(),
        high[2].value(),
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
                p.normal[0].value(),
                p.normal[1].value(),
                p.normal[2].value()
            ),
            crate::geometry::cad::brep::surface::Surface::Cylinder(_) => "cylinder".into(),
            crate::geometry::cad::brep::surface::Surface::Cone(_) => "cone".into(),
            crate::geometry::cad::brep::surface::Surface::Sphere(_) => "sphere".into(),
            crate::geometry::cad::brep::surface::Surface::Torus(_) => "torus".into(),
            _ => "bspline".into(),
        };
        eprintln!(
            "  f{fi:<3} fwd={} {:<28} bbox=[{:.3},{:.3},{:.3}]..[{:.3},{:.3},{:.3}]",
            face.forward as u8, kind, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2],
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
    eprintln!(
        "centre: signed_distance = {:.5}",
        oracle.signed_distance(&centre)
    );
    if let Ok(spec) = std::env::var("STEP_PROBE_POINTS") {
        for chunk in spec.split(';') {
            let c: Vec<f64> = chunk
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if c.len() != 3 {
                continue;
            }
            let p = crate::geometry::Coordinate::from([c[0], c[1], c[2]]);
            let report = oracle.patch_report(&p);
            let (_, kind, d, pt, n) = report.iter().next().cloned().unwrap();
            eprintln!(
                "probe {c:?}: sd={:.7} local_diameter={:.7}  nearest {kind} d={d:.7} at [{:.6},{:.6},{:.6}] n=[{:.4},{:.4},{:.4}]",
                oracle.signed_distance(&p),
                oracle.local_diameter(&p),
                pt[0],
                pt[1],
                pt[2],
                n[0],
                n[1],
                n[2],
            );
            for (index, kind, dist, point, normal) in report.iter().take(6) {
                eprintln!(
                    "    patch #{index:<3} [{kind:<8}] dist {dist:.7} at [{:.6},{:.6},{:.6}] n=[{:.4},{:.4},{:.4}]",
                    point[0], point[1], point[2], normal[0], normal[1], normal[2],
                );
            }
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
                if oracle.signed_distance(&deep) > 0.0 {
                    inside_pos += 1
                } else {
                    inside_neg += 1
                }
                let far = crate::geometry::Coordinate::from([
                    low[0].value() - span[0] * (0.3 + f(i)),
                    low[1].value() + span[1] * f(j),
                    low[2].value() + span[2] * f(k),
                ]);
                if oracle.signed_distance(&far) > 0.0 {
                    outside_pos += 1
                } else {
                    outside_neg += 1
                }
            }
        }
    }
    eprintln!(
        "deep-inside points:  {inside_pos} positive, {inside_neg} negative (want all positive)"
    );
    eprintln!(
        "far-outside points:  {outside_pos} positive, {outside_neg} negative (want all negative)"
    );
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
        let p: [f64; 3] = std::array::from_fn(|k| low[k].value() + span[k] * rand());
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
        let c: Vec<f64> = chunk
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
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

/// For a local `.stp` at `STEP_MESH_FILE`, lists every planar face's trimming
/// loops by their edge-curve kinds and whether `planar_face` accepts them —
/// so the exact loop shape behind a "trimming loop" error can be read off a
/// file this repo cannot see.
#[test]
#[ignore = "audits the planar-face trim loops of STEP_MESH_FILE"]
fn probe_planar_faces() {
    use crate::geometry::cad::brep::{curve::Curve, surface::Surface};

    let Ok(path) = std::env::var("STEP_MESH_FILE") else {
        return;
    };
    let text = std::fs::read_to_string(&path).unwrap();
    let kind = |curve: &Curve| match curve {
        Curve::Line(_) => "Line",
        Curve::Circle(_) => "Circle",
        Curve::Ellipse(_) => "Ellipse",
        Curve::BSpline(_) => "BSpline",
    };
    for (si, brep) in read_all(&text)
        .expect("read failed")
        .into_iter()
        .enumerate()
    {
        let (mut ok, mut err) = (0, 0);
        for (fi, face) in brep.faces.iter().enumerate() {
            if !matches!(face.surface, Surface::Plane(_)) {
                continue;
            }
            let signature = face
                .bounds
                .iter()
                .map(|bound| {
                    let kinds: Vec<&str> = bound
                        .half_edges
                        .iter()
                        .map(|half_edge| kind(&brep.edges[half_edge.edge].curve))
                        .collect();
                    format!("[{}]", kinds.join(","))
                })
                .collect::<Vec<_>>()
                .join(" ");
            match brep.planar_face(face) {
                Ok(_) => ok += 1,
                Err(error) => {
                    err += 1;
                    eprintln!("solid {si} face {fi}: {signature} -> ERR {error}");
                }
            }
        }
        eprintln!("solid {si}: {ok} planar faces ok, {err} failed");
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

/// [`SPHERE_MILLIMETRES`], but a stray micrometre length unit is declared
/// first (as a surface-texture measure would), and the geometry's own
/// representation context assigns the millimetre unit.
const SPHERE_MIXED_UNITS: &str = r#"
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('sphere, mixed units'),'2;1');
FILE_NAME('sphere.step','2026-09-01T00:00:00',(''),(''),'conspire','conspire','');
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
#200 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MICRO.,.METRE.) );
#201 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );
#202 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#201)) REPRESENTATION_CONTEXT('','') );
#210 = LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.),#200);
ENDSEC;
END-ISO-10303-21;
"#;

#[test]
fn the_geometry_context_picks_the_length_unit_over_a_stray_one() {
    use crate::geometry::cad::brep::surface::Surface;

    let brep = read(SPHERE_MIXED_UNITS).unwrap();
    let Surface::Sphere(sphere) = &brep.faces[0].surface else {
        panic!("face is not spherical");
    };
    // The context assigns millimetres: 3000 mm -> 3 m, not 3000 um -> 3 mm.
    assert!((sphere.radius - 3.0).abs() < 1e-9, "{}", sphere.radius);
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

// ---------------------------------------------------------------------------
// Corpus snapshot probes. Both are #[ignore]d and read STEP_CORPUS_DIR (walked
// recursively); with the dir unset they no-op. The stats text is diffed against
// a checked-in snapshot under snapshots/; UPDATE_SNAPSHOT=1 rewrites it. Point
// the dir at ~/Downloads/steptools + ~/Downloads/NIST-PMI-STEP-Files.
// ---------------------------------------------------------------------------

const HEX_FACES: [[usize; 4]; 6] = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
];

fn step_corpus() -> Vec<(String, std::path::PathBuf)> {
    let Ok(root) = std::env::var("STEP_CORPUS_DIR") else {
        return Vec::new();
    };
    let root = std::path::PathBuf::from(root);
    fn walk(
        dir: &std::path::Path,
        root: &std::path::Path,
        out: &mut Vec<(String, std::path::PathBuf)>,
    ) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, root, out);
            } else if path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("stp") || e.eq_ignore_ascii_case("step"))
            {
                let rel = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                out.push((rel, path));
            }
        }
    }
    let mut files = Vec::new();
    walk(&root, &root, &mut files);
    files.sort();
    files
}

fn check_snapshot(name: &str, actual: &str) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/geometry/cad/read/step/brep/snapshots")
        .join(name);
    if std::env::var("UPDATE_SNAPSHOT").is_ok() || !path.exists() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, actual).unwrap();
        eprintln!("wrote snapshot {}", path.display());
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap();
    if expected == actual {
        return;
    }
    let (exp, act): (Vec<&str>, Vec<&str>) = (expected.lines().collect(), actual.lines().collect());
    let mut diff = String::new();
    for row in 0..exp.len().max(act.len()) {
        let (e, a) = (
            exp.get(row).copied().unwrap_or(""),
            act.get(row).copied().unwrap_or(""),
        );
        if e != a {
            diff.push_str(&format!("-{e}\n+{a}\n"));
        }
    }
    panic!("snapshot {name} changed (UPDATE_SNAPSHOT=1 to accept):\n{diff}");
}

/// Parses every corpus file, catching panics, and snapshots per file: the solid
/// and face counts, the surface-type histogram, how many solids reduce to a
/// primitive, how many build an oracle, and the assembly outcome — or the parse
/// error / `PANIC`. A regression here is a reader change that broke a file it
/// used to read (or started panicking on one).
#[test]
#[ignore = "parses every .stp under STEP_CORPUS_DIR and diffs snapshots/corpus_parse.txt"]
fn corpus_parse_snapshot() {
    use crate::geometry::cad::{assemble::assemble, brep::surface::Surface};

    let files = step_corpus();
    if files.is_empty() {
        return;
    }
    let mut report = String::new();
    for (rel, path) in &files {
        let Ok(text) = std::fs::read_to_string(path) else {
            report.push_str(&format!("{rel}: read-io-error\n"));
            continue;
        };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| read_all(&text))) {
            Err(_) => report.push_str(&format!("{rel}: PANIC\n")),
            Ok(Err(error)) => report.push_str(&format!(
                "{rel}: parse-error: {}\n",
                error.to_string().trim()
            )),
            Ok(Ok(breps)) => {
                let faces: usize = breps.iter().map(|brep| brep.faces.len()).sum();
                let mut hist = [0usize; 7];
                for brep in &breps {
                    for face in &brep.faces {
                        hist[match face.surface {
                            Surface::Plane(_) => 0,
                            Surface::Cylinder(_) => 1,
                            Surface::Sphere(_) => 2,
                            Surface::Cone(_) => 3,
                            Surface::Torus(_) => 4,
                            Surface::BSpline(_) => 5,
                            Surface::Revolution(_) => 6,
                        }] += 1;
                    }
                }
                let primitives = breps
                    .iter()
                    .filter(|brep| brep.primitive().is_some())
                    .count();
                let oracles = breps.iter().filter(|brep| brep.oracle().is_ok()).count();
                let assembled = match assemble(&breps) {
                    Ok(bodies) => format!("{} bodies", bodies.len()),
                    Err(error) => format!("no ({error})"),
                };
                report.push_str(&format!(
                    "{rel}: {} solids, {faces} faces \
                     [P{} Cy{} S{} Co{} T{} B{} R{}], {primitives} primitive, \
                     oracle {oracles}/{}, assemble {assembled}\n",
                    breps.len(),
                    hist[0],
                    hist[1],
                    hist[2],
                    hist[3],
                    hist[4],
                    hist[5],
                    hist[6],
                    breps.len(),
                ));
            }
        }
    }
    report.push_str(&format!("\n{} files\n", files.len()));
    check_snapshot("corpus_parse.txt", &report);
}

/// Runs the octree -> dual -> trim pipeline (no fit — the fit is
/// non-deterministic) on every corpus file at a fixed sizing and snapshots the
/// hex count, mesh bounding-box spans, boundary-quad count (with any
/// non-manifold quad flagged), and worst scaled Jacobian — or the pipeline
/// error. Files over `STEP_CORPUS_MAX_BYTES` (default 2 MB) are recorded as
/// `skipped` to keep the default run bounded.
#[test]
#[ignore = "meshes every .stp under STEP_CORPUS_DIR and diffs snapshots/corpus_mesh.txt"]
fn corpus_mesh_snapshot() {
    use crate::{
        geometry::{
            cad::{assemble::assemble, sizing::FeatureSizing},
            mesh::{Connectivity, Mesh, Verdict},
            ntree::Balancing,
            solid::{Solid, Uniform},
        },
        math::{Quantity, Tensor},
        units::Length,
    };

    let files = step_corpus();
    if files.is_empty() {
        return;
    }
    let env_usize = |key, default| -> usize {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    // A single runaway mesh SIGKILLs the whole snapshot, so bound the work up
    // front: skip large files and face-dense solids, and cap the octree. All
    // three are env-overridable for a deliberate full run.
    let max_bytes = env_usize("STEP_CORPUS_MAX_BYTES", 2_000_000) as u64;
    let max_faces = env_usize("STEP_CORPUS_MAX_FACES", 160);
    let levels = Some(env_usize("STEP_CORPUS_LEVELS", 6) as u32);

    let cell = || Quantity::<Length>::new(3.0e-3);
    let fold_bbox = |mesh: &Mesh<3>, low: &mut [f64; 3], high: &mut [f64; 3]| {
        for point in mesh.coordinates().iter() {
            for k in 0..3 {
                low[k] = low[k].min(point[k].value());
                high[k] = high[k].max(point[k].value());
            }
        }
    };
    let faces_of = |mesh: &Mesh<3>| -> (usize, usize) {
        let [Connectivity::Hexahedral(block)] = mesh.connectivities() else {
            return (0, 0);
        };
        let mut seen: std::collections::HashMap<[usize; 4], usize> =
            std::collections::HashMap::new();
        for hex in block.iter() {
            for face in HEX_FACES {
                let mut key = face.map(|corner| hex[corner]);
                key.sort_unstable();
                *seen.entry(key).or_insert(0) += 1;
            }
        }
        (
            seen.values().filter(|&&count| count == 1).count(),
            seen.values().filter(|&&count| count > 2).count(),
        )
    };

    let mut report = String::new();
    for (rel, path) in &files {
        if std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) > max_bytes {
            report.push_str(&format!("{rel}: skipped (over {max_bytes} bytes)\n"));
            continue;
        }
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
            let breps = read_all(&text).map_err(|e| e.to_string())?;
            let face_count: usize = breps.iter().map(|brep| brep.faces.len()).sum();
            if face_count > max_faces {
                return Err(format!("skipped ({face_count} faces)"));
            }
            let (mut elements, mut boundary, mut nonmanifold) = (0usize, 0usize, 0usize);
            let mut worst = f64::INFINITY;
            let mut low = [f64::INFINITY; 3];
            let mut high = [f64::NEG_INFINITY; 3];
            let mut tally = |mesh: &Mesh<3>| {
                elements += mesh.number_of_elements();
                worst = worst.min(
                    mesh.minimum_scaled_jacobians()[0]
                        .iter()
                        .copied()
                        .fold(f64::INFINITY, f64::min),
                );
                let (b, n) = faces_of(mesh);
                boundary += b;
                nonmanifold += n;
                fold_bbox(mesh, &mut low, &mut high);
            };
            if let Ok(bodies) = assemble(&breps) {
                let sizing = Uniform(cell());
                for body in &bodies {
                    let (mesh, _) = body
                        .trim(&sizing, levels, 0.1, Balancing::Strong(1))
                        .map_err(|e| e.to_string())?;
                    tally(&mesh);
                }
            } else {
                for brep in &breps {
                    brep.oracle().map_err(|e| e.to_string())?;
                    let sizing = FeatureSizing::of(
                        brep,
                        32,
                        Quantity::<Length>::new(4.0e-4),
                        Some(cell()),
                        Some(0.2),
                    );
                    let (mesh, _) = brep
                        .trim(&sizing, levels, 0.1, Balancing::Strong(1))
                        .map_err(|e| e.to_string())?;
                    tally(&mesh);
                }
            }
            Ok::<_, String>((elements, boundary, nonmanifold, worst, low, high))
        }));
        match outcome {
            Err(_) => report.push_str(&format!("{rel}: PANIC\n")),
            Ok(Err(error)) if error.starts_with("skipped") => {
                report.push_str(&format!("{rel}: {error}\n"))
            }
            Ok(Err(error)) => report.push_str(&format!("{rel}: mesh-error: {error}\n")),
            Ok(Ok((elements, boundary, nonmanifold, worst, low, high))) => {
                let span = |k: usize| (high[k] - low[k]).max(0.0);
                let flag = if nonmanifold > 0 {
                    format!(", {nonmanifold} non-manifold")
                } else {
                    String::new()
                };
                report.push_str(&format!(
                    "{rel}: {elements} hexes, bbox [{:.4} {:.4} {:.4}], \
                     {boundary} boundary faces{flag}, worst SJ {:.3}\n",
                    span(0),
                    span(1),
                    span(2),
                    worst,
                ));
            }
        }
    }
    report.push_str(&format!("\n{} files\n", files.len()));
    check_snapshot("corpus_mesh.txt", &report);
}

/// Meshes the `STEP_MESH_FILE` solid with the same feature-sizing knobs as
/// `probe_mesh_real_file` (`STEP_MESH_MIN`/`_CELL`/`_SEGMENTS`/`_GRADATION`/
/// `_PROXIMITY`/`_CURVATURE`, plus `STEP_MESH_CREASE`+`STEP_MESH_CREASE_CELLS`)
/// then prints **where** the poor hexes are: every element whose minimum
/// scaled Jacobian is at or below `STEP_INVERT_SJ` (default 0.0 -> only
/// inverted), reported as a centroid and edge scale, followed by the bounding
/// box of that set. If the tangle is a buffer-fit inversion on a specific
/// micro-feature, the bad hexes cluster there; a reader/reconstruction bug
/// would smear them across the whole face instead.
#[test]
#[ignore = "locates inverted hexes for STEP_MESH_FILE (SJ <= STEP_INVERT_SJ)"]
fn probe_inverted_hexes() {
    use crate::{
        geometry::{
            cad::sizing::FeatureSizing,
            mesh::{Connectivity, Fitting, Verdict},
            ntree::Balancing,
            solid::Solid,
        },
        math::Quantity,
        units::Length,
    };
    let Ok(path) = std::env::var("STEP_MESH_FILE") else {
        return;
    };
    let env_f64 = |key, default: f64| -> f64 {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let length = |v| Quantity::<Length>::new(v);
    let text = std::fs::read_to_string(&path).unwrap();
    let breps = read_all(&text).expect("read failed");
    let levels = std::env::var("STEP_MESH_LEVELS")
        .ok()
        .and_then(|v| v.parse().ok());
    let cell = env_f64("STEP_MESH_CELL", 6.0e-3);
    let maximum =
        (std::env::var("STEP_MESH_CELL").as_deref() != Ok("none")).then(|| length(cell));
    let gradation = match std::env::var("STEP_MESH_GRADATION").as_deref() {
        Ok("none") => None,
        Ok(v) => Some(v.parse().expect("STEP_MESH_GRADATION")),
        Err(_) => Some(0.2),
    };
    let threshold = env_f64("STEP_INVERT_SJ", 0.0);

    for (index, brep) in breps.iter().enumerate() {
        eprintln!("--- solid {index}: {} faces ---", brep.faces.len());
        let mut sizing = FeatureSizing::of(
            brep,
            env_f64("STEP_MESH_SEGMENTS", 24.0) as usize,
            length(env_f64("STEP_MESH_MIN", cell / 8.0)),
            maximum,
            gradation,
        );
        if let Ok(n) = std::env::var("STEP_MESH_PROXIMITY") {
            sizing = sizing
                .with_proximity(brep, n.parse().expect("STEP_MESH_PROXIMITY"))
                .expect("with_proximity");
        }
        if let Ok(n) = std::env::var("STEP_MESH_CURVATURE") {
            sizing = sizing
                .with_curvature(brep, n.parse().expect("STEP_MESH_CURVATURE"))
                .expect("with_curvature");
        }
        if let Ok(r) = std::env::var("STEP_MESH_CREASE") {
            let cells = std::env::var("STEP_MESH_CREASE_CELLS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1);
            let hops = std::env::var("STEP_MESH_CREASE_HOPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1);
            sizing = sizing
                .with_crease_proximity(brep, length(r.parse().expect("STEP_MESH_CREASE")), cells, hops)
                .expect("with_crease_proximity");
        }
        let mesh = brep
            .mesh(&sizing, levels, 0.1, Balancing::Strong(1), Fitting::Soft)
            .expect("mesh failed");
        let coords = mesh.coordinates();
        let sj = mesh.minimum_scaled_jacobians();
        let [Connectivity::Hexahedral(block)] = mesh.connectivities() else {
            eprintln!("  not a single hex block");
            continue;
        };
        let mut bad = 0usize;
        let mut low = [f64::INFINITY; 3];
        let mut high = [f64::NEG_INFINITY; 3];
        let mut worst = f64::INFINITY;
        let mut worst_c = [0.0; 3];
        let bin = env_f64("STEP_INVERT_BIN", 2.0e-4);
        let mut bins: std::collections::HashMap<[i64; 3], (usize, [f64; 3], f64)> =
            std::collections::HashMap::new();
        for (element, hex) in block.iter().enumerate() {
            let j = sj[0][element];
            worst = worst.min(j);
            if !(j <= threshold) {
                continue;
            }
            bad += 1;
            let centroid: [f64; 3] = std::array::from_fn(|k| {
                hex.iter().map(|&n| coords[n][k].value()).sum::<f64>() / 8.0
            });
            for k in 0..3 {
                low[k] = low[k].min(centroid[k]);
                high[k] = high[k].max(centroid[k]);
            }
            let key = std::array::from_fn(|k| (centroid[k] / bin).floor() as i64);
            let entry = bins.entry(key).or_insert((0, [0.0; 3], 0.0));
            entry.0 += 1;
            for k in 0..3 {
                entry.1[k] += centroid[k];
            }
            entry.2 = entry.2.min(j);
            // Edge scale: mean of the three axis spans of the hex's nodes.
            let (mut elo, mut ehi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for &n in hex.iter() {
                for k in 0..3 {
                    elo[k] = elo[k].min(coords[n][k].value());
                    ehi[k] = ehi[k].max(coords[n][k].value());
                }
            }
            let scale = (0..3).map(|k| ehi[k] - elo[k]).sum::<f64>() / 3.0;
            if j <= worst + 1e-12 {
                worst_c = centroid;
            }
            if bad <= env_f64("STEP_INVERT_LIST", 40.0) as usize {
                eprintln!(
                    "  bad hex: SJ {j:+.4}  centroid [{:.5} {:.5} {:.5}]  edge ~{scale:.6}",
                    centroid[0], centroid[1], centroid[2],
                );
            }
        }
        eprintln!(
            "  {bad} / {} hexes with SJ <= {threshold}; worst SJ {worst:.4} near [{:.5} {:.5} {:.5}]",
            mesh.number_of_elements(),
            worst_c[0], worst_c[1], worst_c[2],
        );
        if bad > 0 {
            eprintln!(
                "  bad-hex centroid bbox: x [{:.5} {:.5}]  y [{:.5} {:.5}]  z [{:.5} {:.5}]",
                low[0], high[0], low[1], high[1], low[2], high[2],
            );
            let mut clusters: Vec<_> = bins
                .values()
                .map(|(n, sum, wj)| {
                    (*n, std::array::from_fn::<f64, 3, _>(|k| sum[k] / *n as f64), *wj)
                })
                .collect();
            clusters.sort_by(|a, b| b.0.cmp(&a.0));
            eprintln!("  fold clusters (bin {bin:.0e}m):");
            for (n, c, wj) in clusters.iter().take(10) {
                eprintln!(
                    "      {n:4} bad near [{:.5} {:.5} {:.5}]  worst SJ {wj:+.3}",
                    c[0], c[1], c[2]
                );
            }
        }
    }
}

/// Global hole census on the **pre-fit** trimmed mesh (`brep.trim`). A "hole" is
/// a boundary face of the kept mesh (a face owned by exactly one kept cell)
/// whose centroid sits strictly **inside** the solid (signed distance
/// `> STEP_HOLE_DEPTH`, default 25um) — a face that should have had a neighbour
/// cell but doesn't, i.e. a missing element. Scans the whole mesh and bins the
/// offending faces into clusters (`STEP_HOLE_BIN`, default 5e-4 m). For the
/// deepest hole it dumps the oracle's ray-parity per direction so a classifier
/// false-`Outside` (the flip-side risk of the spur agreement-voting fix) is
/// visible as a should-be-inside point the vote dropped.
#[test]
#[ignore = "global interior-hole census on the trimmed mesh for STEP_MESH_FILE"]
fn probe_holes() {
    use crate::{
        geometry::{
            Coordinate,
            cad::sizing::FeatureSizing,
            mesh::Connectivity,
            ntree::Balancing,
            solid::{Solid, SolidOracle},
        },
        math::Quantity,
        units::Length,
    };
    let Ok(path) = std::env::var("STEP_MESH_FILE") else {
        return;
    };
    let env_f64 = |key, default: f64| -> f64 {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let length = |v| Quantity::<Length>::new(v);
    let text = std::fs::read_to_string(&path).unwrap();
    let breps = read_all(&text).expect("read failed");
    let levels = std::env::var("STEP_MESH_LEVELS")
        .ok()
        .and_then(|v| v.parse().ok());
    let cell = env_f64("STEP_MESH_CELL", 6.0e-3);
    let maximum =
        (std::env::var("STEP_MESH_CELL").as_deref() != Ok("none")).then(|| length(cell));
    let gradation = match std::env::var("STEP_MESH_GRADATION").as_deref() {
        Ok("none") => None,
        Ok(v) => Some(v.parse().expect("STEP_MESH_GRADATION")),
        Err(_) => Some(0.2),
    };
    let depth = env_f64("STEP_HOLE_DEPTH", 2.5e-5);
    let bin = env_f64("STEP_HOLE_BIN", 5.0e-4);

    for (index, brep) in breps.iter().enumerate() {
        eprintln!("--- solid {index}: {} faces ---", brep.faces.len());
        let mut sizing = FeatureSizing::of(
            brep,
            env_f64("STEP_MESH_SEGMENTS", 24.0) as usize,
            length(env_f64("STEP_MESH_MIN", cell / 8.0)),
            maximum,
            gradation,
        );
        if let Ok(n) = std::env::var("STEP_MESH_PROXIMITY") {
            sizing = sizing.with_proximity(brep, n.parse().unwrap()).unwrap();
        }
        if let Ok(n) = std::env::var("STEP_MESH_CURVATURE") {
            sizing = sizing.with_curvature(brep, n.parse().unwrap()).unwrap();
        }

        let (mesh, _) = brep
            .trim(&sizing, levels, 0.1, Balancing::Strong(1))
            .expect("trim failed");
        let coords = mesh.coordinates();
        let [Connectivity::Hexahedral(block)] = mesh.connectivities() else {
            eprintln!("  not a single hex block");
            continue;
        };
        let oracle = brep.oracle().expect("oracle");

        const FACES: [[usize; 4]; 6] = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ];
        // Count owners per face so boundary faces (exactly one owner) are known.
        let mut face_count: std::collections::HashMap<[usize; 4], usize> =
            std::collections::HashMap::new();
        for hex in block.iter() {
            for f in FACES {
                let mut key = [hex[f[0]], hex[f[1]], hex[f[2]], hex[f[3]]];
                key.sort_unstable();
                *face_count.entry(key).or_insert(0) += 1;
            }
        }

        // Scan boundary faces; a hole face has its centroid strictly inside.
        let mut holes = 0usize;
        let mut bins: std::collections::HashMap<[i64; 3], (usize, [f64; 3], f64)> =
            std::collections::HashMap::new();
        let mut deepest = (0.0f64, [0.0f64; 3]);
        for hex in block.iter() {
            for f in FACES {
                let mut key = [hex[f[0]], hex[f[1]], hex[f[2]], hex[f[3]]];
                key.sort_unstable();
                if face_count.get(&key).copied().unwrap_or(0) != 1 {
                    continue;
                }
                let fc: [f64; 3] = std::array::from_fn(|k| {
                    f.iter().map(|&i| coords[hex[i]][k].value()).sum::<f64>() / 4.0
                });
                let sd = oracle.signed_distance(&Coordinate::from(fc));
                if sd <= depth {
                    continue; // on/near the true boundary -> legitimate face
                }
                holes += 1;
                let b = std::array::from_fn(|k| (fc[k] / bin).floor() as i64);
                let e = bins.entry(b).or_insert((0, [0.0; 3], 0.0));
                e.0 += 1;
                for k in 0..3 {
                    e.1[k] += fc[k];
                }
                e.2 = e.2.max(sd);
                if sd > deepest.0 {
                    deepest = (sd, fc);
                }
            }
        }
        eprintln!(
            "  {holes} interior-hole faces (centroid inside by > {depth:.1e} m) over {} kept cells",
            mesh.number_of_elements()
        );
        if holes == 0 {
            continue;
        }
        let mut clusters: Vec<_> = bins
            .values()
            .map(|(n, s, d)| (*n, std::array::from_fn::<f64, 3, _>(|k| s[k] / *n as f64), *d))
            .collect();
        clusters.sort_by(|a, b| b.0.cmp(&a.0));
        eprintln!("  hole clusters (bin {bin:.0e} m):");
        for (n, c, d) in clusters.iter().take(12) {
            eprintln!(
                "      {n:5} faces near [{:.5} {:.5} {:.5}]  deepest inside {d:.6}",
                c[0], c[1], c[2]
            );
        }
        eprintln!(
            "  deepest hole face at [{:.5} {:.5} {:.5}], inside by {:.6}",
            deepest.1[0], deepest.1[1], deepest.1[2], deepest.0
        );
        // Probe the oracle right at the deepest hole: is a should-be-inside
        // point being classified Outside by the ray-parity vote? Dump the per-
        // direction crossing parity for the three fixed directions; an even
        // (outside) parity here on a point the SDF says is inside is a
        // classifier false-`Outside`.
        let p = Coordinate::from(deepest.1);
        eprintln!(
            "  oracle at deepest hole: signed {:.9} distance {:.9} (positive => inside) p=[{:.9} {:.9} {:.9}]",
            oracle.signed_distance(&p),
            oracle.distance(&p),
            deepest.1[0], deepest.1[1], deepest.1[2],
        );
        let dirs = [
            [0.862_667, 0.411_988, 0.291_536],
            [0.301_511, 0.904_534, 0.301_511],
            [0.334_412, 0.243_975, 0.910_367],
        ];
        for dir in dirs {
            let report = oracle.ray_report(&p, dir);
            eprintln!(
                "    dir {dir:?}: {} hits (parity {} => {})",
                report.len(),
                report.len() % 2,
                if report.len() % 2 == 1 { "inside" } else { "outside" },
            );
            for (patch, kind, t) in &report {
                eprintln!("        patch #{patch} [{kind}] t = {t:.8}");
            }
        }
        eprintln!("  nearest patches at deepest hole:");
        for (index, kind, dist, point, normal) in oracle.patch_report(&p).into_iter().take(6) {
            let diff = [deepest.1[0] - point[0], deepest.1[1] - point[1], deepest.1[2] - point[2]];
            let dot = diff[0] * normal[0] + diff[1] * normal[1] + diff[2] * normal[2];
            eprintln!(
                "      patch #{index} [{kind}] dist {dist:.9} at [{:.9} {:.9} {:.9}] n [{:.4} {:.4} {:.4}] (q-p).n={dot:+.9}",
                point[0], point[1], point[2], normal[0], normal[1], normal[2]
            );
        }
        let (best, boxes) = oracle.nearest_report(&p);
        eprintln!("  nearest() itself: {best:?}");
        let mut boxes_sorted = boxes.clone();
        boxes_sorted.sort_by(|a, b| a.2.total_cmp(&b.2));
        eprintln!("  patch boxes nearest first:");
        for (index, kind, boxdist) in boxes_sorted.iter().take(8) {
            eprintln!("      patch #{index} [{kind}] box-distance {boxdist:.9}");
        }
        eprintln!("  graze sweep on dir0:");
        for floor in [0.0, 1e-8, 1e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4] {
            let rep = oracle.ray_report_grazed(&p, dirs[0], floor);
            let flags: Vec<String> = rep
                .iter()
                .map(|(patch, _, _, g)| format!("#{patch}{}", if *g { "*" } else { "" }))
                .collect();
            eprintln!("    floor {floor:.0e}: {} hits [{}]", rep.len(), flags.join(" "));
        }
    }
}

/// Trims the `STEP_MESH_FILE` solid (dual + classify, no fit) and hunts the
/// **spur**: kept (`Inside`/`Cut`) cells whose centroid lies more than
/// `STEP_SPUR_MARGIN` (default 0.2mm, in metres) outside the tight bounding
/// box of the solid's own vertices. A phantom diagonal escaping the part is a
/// ray-parity misclassification, so for the spur cell farthest out we dump the
/// oracle's ray crossings along all three parity directions and the nearest
/// patches, naming the surface whose hit count is wrong.
#[test]
#[ignore = "locates trim spur cells for STEP_MESH_FILE and interrogates the oracle"]
fn probe_trim_spur() {
    use crate::{
        geometry::{
            Coordinate,
            cad::sizing::FeatureSizing,
            mesh::{Class, Connectivity},
            ntree::Balancing,
            solid::{Solid, SolidOracle},
        },
        math::Quantity,
        units::Length,
    };
    let Ok(path) = std::env::var("STEP_MESH_FILE") else {
        return;
    };
    let env_f64 = |key, default: f64| -> f64 {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let length = |v| Quantity::<Length>::new(v);
    let text = std::fs::read_to_string(&path).unwrap();
    let breps = read_all(&text).expect("read failed");
    let levels = std::env::var("STEP_MESH_LEVELS")
        .ok()
        .and_then(|v| v.parse().ok());
    let cell = env_f64("STEP_MESH_CELL", 6.0e-3);
    let maximum =
        (std::env::var("STEP_MESH_CELL").as_deref() != Ok("none")).then(|| length(cell));
    let gradation = match std::env::var("STEP_MESH_GRADATION").as_deref() {
        Ok("none") => None,
        Ok(v) => Some(v.parse().expect("STEP_MESH_GRADATION")),
        Err(_) => Some(0.2),
    };
    let margin = env_f64("STEP_SPUR_MARGIN", 2.0e-4);

    for (index, brep) in breps.iter().enumerate() {
        eprintln!("--- solid {index}: {} faces ---", brep.faces.len());
        // Tight bbox of the B-rep's own vertices (the true part extent).
        let mut vlo = [f64::INFINITY; 3];
        let mut vhi = [f64::NEG_INFINITY; 3];
        for v in &brep.vertices {
            for k in 0..3 {
                vlo[k] = vlo[k].min(v[k].value());
                vhi[k] = vhi[k].max(v[k].value());
            }
        }
        eprintln!(
            "  vertex bbox: x [{:.5} {:.5}] y [{:.5} {:.5}] z [{:.5} {:.5}]",
            vlo[0], vhi[0], vlo[1], vhi[1], vlo[2], vhi[2]
        );

        let mut sizing = FeatureSizing::of(
            brep,
            env_f64("STEP_MESH_SEGMENTS", 24.0) as usize,
            length(env_f64("STEP_MESH_MIN", cell / 8.0)),
            maximum,
            gradation,
        );
        if let Ok(n) = std::env::var("STEP_MESH_PROXIMITY") {
            sizing = sizing.with_proximity(brep, n.parse().unwrap()).unwrap();
        }
        if let Ok(n) = std::env::var("STEP_MESH_CURVATURE") {
            sizing = sizing.with_curvature(brep, n.parse().unwrap()).unwrap();
        }

        let (mesh, classes) = brep
            .trim(&sizing, levels, 0.1, Balancing::Strong(1))
            .expect("trim failed");
        let coords = mesh.coordinates();
        let [Connectivity::Hexahedral(block)] = mesh.connectivities() else {
            eprintln!("  not a single hex block");
            continue;
        };

        let mut spur = 0usize;
        let mut worst_out = 0.0;
        let mut worst_centroid = [0.0; 3];
        let mut worst_elem = usize::MAX;
        let mut slo = [f64::INFINITY; 3];
        let mut shi = [f64::NEG_INFINITY; 3];
        for (element, hex) in block.iter().enumerate() {
            if classes[element] == Class::Outside {
                continue;
            }
            let c: [f64; 3] =
                std::array::from_fn(|k| hex.iter().map(|&n| coords[n][k].value()).sum::<f64>() / 8.0);
            // How far the centroid pokes past the vertex bbox, any axis.
            let out = (0..3)
                .map(|k| (vlo[k] - c[k]).max(c[k] - vhi[k]).max(0.0))
                .fold(0.0_f64, f64::max);
            if out > margin {
                spur += 1;
                for k in 0..3 {
                    slo[k] = slo[k].min(c[k]);
                    shi[k] = shi[k].max(c[k]);
                }
                if out > worst_out {
                    worst_out = out;
                    worst_centroid = c;
                    worst_elem = element;
                }
            }
        }
        eprintln!(
            "  {spur} kept cells > {margin} outside vertex bbox (of {} kept)",
            mesh.number_of_elements()
        );
        if spur == 0 {
            continue;
        }
        eprintln!(
            "  spur centroid bbox: x [{:.5} {:.5}] y [{:.5} {:.5}] z [{:.5} {:.5}]",
            slo[0], shi[0], slo[1], shi[1], slo[2], shi[2]
        );
        eprintln!(
            "  farthest spur cell: [{:.5} {:.5} {:.5}] pokes {worst_out:.5} out; class {:?}",
            worst_centroid[0], worst_centroid[1], worst_centroid[2], classes[worst_elem],
        );

        // Per-corner: the oracle's ray-parity verdict and signed distance at
        // each of the eight nodes. The flood seeds `Outside` only where all
        // eight read outside; a single spuriously-inside corner keeps the cell.
        let oracle = brep.oracle().expect("oracle");
        let hex: Vec<usize> = block.iter().nth(worst_elem).unwrap().to_vec();
        eprintln!("  eight corners (enclose / signed distance):");
        let dirs = [
            [0.862_667, 0.411_988, 0.291_536],
            [0.301_511, 0.904_534, 0.301_511],
            [0.334_412, 0.243_975, 0.910_367],
        ];
        let mut positive_node = None;
        for &n in &hex {
            let p = Coordinate::from(std::array::from_fn::<f64, 3, _>(|k| coords[n][k].value()));
            let sd = oracle.signed_distance(&p);
            eprintln!(
                "      [{:.5} {:.5} {:.5}]  signed {sd:+.6}",
                coords[n][0].value(),
                coords[n][1].value(),
                coords[n][2].value(),
            );
            if sd > 0.0 {
                positive_node = Some(n);
            }
        }

        // The false-positive corner: dump the raw ray crossings for each of the
        // three parity directions, sorted by t, so a grazed / doubled / missed
        // hit that flips the parity vote is visible.
        if let Some(n) = positive_node {
            let p = Coordinate::from(std::array::from_fn::<f64, 3, _>(|k| coords[n][k].value()));
            eprintln!(
                "  false-positive corner [{:.6} {:.6} {:.6}] ray crossings:",
                coords[n][0].value(),
                coords[n][1].value(),
                coords[n][2].value(),
            );
            for dir in dirs {
                let mut report = oracle.ray_report(&p, dir);
                report.sort_by(|a, b| a.2.total_cmp(&b.2));
                eprintln!("    dir {dir:?}: {} hits (parity {})", report.len(), report.len() % 2);
                for (patch, kind, t) in &report {
                    eprintln!("        patch #{patch} [{kind}] t = {t:.8}");
                }
            }
            // For the first direction, sweep graze floors so the boundary
            // proximity at which each hit is flagged degenerate is visible.
            eprintln!("  graze sweep on dir {:?}:", dirs[0]);
            for floor in [0.0, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 2e-4, 5e-4] {
                let rep = oracle.ray_report_grazed(&p, dirs[0], floor);
                let flags: Vec<String> = rep
                    .iter()
                    .map(|(patch, _, _, g)| format!("#{patch}{}", if *g { "*" } else { "" }))
                    .collect();
                eprintln!(
                    "    floor {floor:.0e}: {} hits [{}]",
                    rep.len(),
                    flags.join(" ")
                );
            }
        }

        // Interrogate the oracle at the centroid: ray crossings per direction,
        // and the nearest patches.
        let q = Coordinate::from(worst_centroid);
        for dir in dirs {
            let report = oracle.ray_report(&q, dir);
            eprintln!("  centroid ray {dir:?}: {} hits", report.len());
            for (patch, kind, t) in &report {
                eprintln!("      patch #{patch} [{kind}] t = {t:.6}");
            }
        }
        eprintln!("  nearest patches to the false-positive corner:");
        let probe = positive_node
            .map(|n| Coordinate::from(std::array::from_fn::<f64, 3, _>(|k| coords[n][k].value())))
            .unwrap_or(q);
        for (kind, dist, point, normal) in oracle.patch_report(&probe).into_iter().map(|(_, k, d, pt, n)| (k, d, pt, n)).take(8) {
            eprintln!(
                "      [{kind}] dist {dist:.6} at [{:.5} {:.5} {:.5}] n [{:.3} {:.3} {:.3}]",
                point[0], point[1], point[2], normal[0], normal[1], normal[2]
            );
        }
    }
}

#[test]
#[ignore]
/// Scan every kept `Cut` cell for the corner-threading signature — a lopsided
/// split where a lone corner disagrees in sign with the other seven. A genuine
/// boundary cut has a balanced split (a face passing through); a lone minority
/// corner is the classic ray-parity false-positive. Reports the census, then
/// clusters the offenders by an axis-aligned grid so the on-part hot spots
/// (e.g. the flange) are visible even though nothing pokes outside the bbox.
fn probe_lopsided_cut() {
    use crate::{
        geometry::{
            Coordinate,
            cad::sizing::FeatureSizing,
            mesh::{Class, Connectivity},
            ntree::Balancing,
            solid::{Solid, SolidOracle},
        },
        math::Quantity,
        units::Length,
    };
    let Ok(path) = std::env::var("STEP_MESH_FILE") else {
        return;
    };
    let env_f64 = |key, default: f64| -> f64 {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let length = |v| Quantity::<Length>::new(v);
    let text = std::fs::read_to_string(&path).unwrap();
    let breps = read_all(&text).expect("read failed");
    let levels = std::env::var("STEP_MESH_LEVELS")
        .ok()
        .and_then(|v| v.parse().ok());
    let cell = env_f64("STEP_MESH_CELL", 6.0e-3);
    let maximum =
        (std::env::var("STEP_MESH_CELL").as_deref() != Ok("none")).then(|| length(cell));
    let gradation = match std::env::var("STEP_MESH_GRADATION").as_deref() {
        Ok("none") => None,
        Ok(v) => Some(v.parse().expect("STEP_MESH_GRADATION")),
        Err(_) => Some(0.2),
    };
    // Cell size to bin offenders into clusters for the census (default 0.2mm).
    let bin = env_f64("STEP_LOPSIDED_BIN", 2.0e-4);

    for (index, brep) in breps.iter().enumerate() {
        eprintln!("--- solid {index}: {} faces ---", brep.faces.len());
        let mut sizing = FeatureSizing::of(
            brep,
            env_f64("STEP_MESH_SEGMENTS", 24.0) as usize,
            length(env_f64("STEP_MESH_MIN", cell / 8.0)),
            maximum,
            gradation,
        );
        if let Ok(n) = std::env::var("STEP_MESH_PROXIMITY") {
            sizing = sizing.with_proximity(brep, n.parse().unwrap()).unwrap();
        }
        if let Ok(n) = std::env::var("STEP_MESH_CURVATURE") {
            sizing = sizing.with_curvature(brep, n.parse().unwrap()).unwrap();
        }

        let (mesh, classes) = brep
            .trim(&sizing, levels, 0.1, Balancing::Strong(1))
            .expect("trim failed");
        let coords = mesh.coordinates();
        let [Connectivity::Hexahedral(block)] = mesh.connectivities() else {
            eprintln!("  not a single hex block");
            continue;
        };
        let oracle = brep.oracle().expect("oracle");

        // Precompute the signed distance at every used node once.
        let mut signed: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        let mut sign_of = |n: usize| -> f64 {
            *signed.entry(n).or_insert_with(|| {
                let p =
                    Coordinate::from(std::array::from_fn::<f64, 3, _>(|k| coords[n][k].value()));
                oracle.signed_distance(&p)
            })
        };

        let mut lopsided: Vec<([f64; 3], usize)> = Vec::new();
        for (element, hex) in block.iter().enumerate() {
            if classes[element] != Class::Cut {
                continue;
            }
            let positives = hex.iter().filter(|&&n| sign_of(n) > 0.0).count();
            // A lone-minority corner (1 in / 7 out, or 7 in / 1 out) is the
            // corner-threading false-positive signature.
            if positives == 1 || positives == 7 {
                let c: [f64; 3] = std::array::from_fn(|k| {
                    hex.iter().map(|&n| coords[n][k].value()).sum::<f64>() / 8.0
                });
                lopsided.push((c, positives));
            }
        }
        let cut = classes.iter().filter(|&&c| c == Class::Cut).count();
        eprintln!(
            "  {} lopsided (1/7) Cut cells of {cut} Cut ({} kept total)",
            lopsided.len(),
            mesh.number_of_elements(),
        );
        if lopsided.is_empty() {
            continue;
        }

        // Bin offenders into a coarse grid and report the densest clusters.
        let mut bins: std::collections::HashMap<[i64; 3], (usize, [f64; 3])> =
            std::collections::HashMap::new();
        for (c, _) in &lopsided {
            let key = std::array::from_fn(|k| (c[k] / bin).floor() as i64);
            let entry = bins.entry(key).or_insert((0, [0.0; 3]));
            entry.0 += 1;
            for k in 0..3 {
                entry.1[k] += c[k];
            }
        }
        let mut clusters: Vec<_> = bins
            .values()
            .map(|(n, sum)| (*n, std::array::from_fn::<f64, 3, _>(|k| sum[k] / *n as f64)))
            .collect();
        clusters.sort_by(|a, b| b.0.cmp(&a.0));
        eprintln!("  top clusters (bin {bin:.0e}m):");
        for (n, c) in clusters.iter().take(12) {
            eprintln!("      {n:4} cells near [{:.5} {:.5} {:.5}]", c[0], c[1], c[2]);
        }

        // Dive on the densest cluster: for a handful of its lopsided cells, dump
        // the lone-minority corner, its signed distance and nearest patch, and
        // how many rays `encloses` needed — a false positive threads a corner
        // (many rays, tiny |signed|, nearest patch far), a real cut sits on a
        // face (clean, |signed| ~ the sliver thickness, nearest patch touching).
        let (_, hot) = clusters[0];
        eprintln!("  dive on densest cluster near [{:.5} {:.5} {:.5}]:", hot[0], hot[1], hot[2]);
        let mut shown = 0;
        for (element, hex) in block.iter().enumerate() {
            if shown >= 6 || classes[element] != Class::Cut {
                continue;
            }
            let c: [f64; 3] = std::array::from_fn(|k| {
                hex.iter().map(|&n| coords[n][k].value()).sum::<f64>() / 8.0
            });
            let near_hot = (0..3).all(|k| (c[k] - hot[k]).abs() < bin);
            if !near_hot {
                continue;
            }
            let positives = hex.iter().filter(|&&n| sign_of(n) > 0.0).count();
            if positives != 1 && positives != 7 {
                continue;
            }
            let minority = if positives == 1 { 1 } else { 7 };
            let lone = *hex
                .iter()
                .find(|&&n| (sign_of(n) > 0.0) == (minority == 1 && positives == 1))
                .unwrap_or(&hex[0]);
            let p = Coordinate::from(std::array::from_fn::<f64, 3, _>(|k| coords[lone][k].value()));
            let sd = oracle.signed_distance(&p);
            let nearest = oracle.patch_report(&p).into_iter().next();
            eprintln!(
                "    cell {element} pos={positives} lone [{:.5} {:.5} {:.5}] signed {sd:+.6}{}",
                coords[lone][0].value(),
                coords[lone][1].value(),
                coords[lone][2].value(),
                nearest
                    .map(|(_, k, d, _, _)| format!("  nearest {k} @ {d:.6}"))
                    .unwrap_or_default(),
            );
            shown += 1;
        }
    }
}

/// Lists every crease edge (`Brep::features().creases`) of `STEP_MESH_FILE`
/// with its chord length and world-space endpoints/midpoint, sorted longest
/// first — locates a specific crease (e.g. "the long one behind the radius
/// below the flange") without opening a viewer, and lets a per-crease damage
/// census (see `probe_crease_damage`) be pointed at the right one.
#[test]
#[ignore = "lists STEP_MESH_FILE's crease edges by chord length"]
fn probe_creases() {
    let Ok(path) = std::env::var("STEP_MESH_FILE") else {
        return;
    };
    let text = std::fs::read_to_string(&path).unwrap();
    let breps = read_all(&text).expect("read failed");
    for (index, brep) in breps.iter().enumerate() {
        let features = brep.features();
        eprintln!(
            "--- solid {index}: {} faces, {} creases, {} corners ---",
            brep.faces.len(),
            features.creases.len(),
            features.corners.len(),
        );
        let mut rows: Vec<(usize, f64, [f64; 3], [f64; 3], [f64; 3])> = features
            .creases
            .iter()
            .map(|&edge| {
                let [ia, ib] = brep.edges[edge].vertices;
                let a: [f64; 3] = std::array::from_fn(|k| brep.vertices[ia][k].value());
                let b: [f64; 3] = std::array::from_fn(|k| brep.vertices[ib][k].value());
                let mid: [f64; 3] = std::array::from_fn(|k| 0.5 * (a[k] + b[k]));
                let length = (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt();
                (edge, length, a, b, mid)
            })
            .collect();
        rows.sort_by(|x, y| y.1.total_cmp(&x.1));
        for (edge, length, a, b, mid) in rows.iter().take(30) {
            eprintln!(
                "  edge #{edge}: chord {length:.6}  a=[{:.5} {:.5} {:.5}]  b=[{:.5} {:.5} {:.5}]  mid=[{:.5} {:.5} {:.5}]",
                a[0], a[1], a[2], b[0], b[1], b[2], mid[0], mid[1], mid[2],
            );
        }
        // Region filter: STEP_CREASE_NEAR="x,y,z,radius" prints every crease
        // (not just the longest 30) whose midpoint or either endpoint falls
        // within radius of the given point, so a crease matching a known
        // damage cluster (e.g. the bad-hex centroid bbox from
        // `probe_inverted_hexes`) can be picked out directly.
        if let Ok(spec) = std::env::var("STEP_CREASE_NEAR") {
            let parts: Vec<f64> = spec.split(',').filter_map(|s| s.parse().ok()).collect();
            if parts.len() == 4 {
                let (px, py, pz, radius) = (parts[0], parts[1], parts[2], parts[3]);
                let near = |p: [f64; 3]| {
                    ((p[0] - px).powi(2) + (p[1] - py).powi(2) + (p[2] - pz).powi(2)).sqrt()
                        <= radius
                };
                eprintln!("  creases within {radius} of [{px} {py} {pz}]:");
                for (edge, length, a, b, mid) in &rows {
                    if near(*a) || near(*b) || near(*mid) {
                        eprintln!(
                            "    edge #{edge}: chord {length:.6}  a=[{:.5} {:.5} {:.5}]  b=[{:.5} {:.5} {:.5}]  mid=[{:.5} {:.5} {:.5}]",
                            a[0], a[1], a[2], b[0], b[1], b[2], mid[0], mid[1], mid[2],
                        );
                    }
                }
            }
        }
        // STEP_CREASE_EDGE=N: names the two faces incident to brep edge #N
        // (as printed above) and their surface kind, so a specific crease
        // (once located above) can be matched to "the fillet behind the
        // radius" by eye.
        if let Ok(spec) = std::env::var("STEP_CREASE_EDGE") {
            let edge: usize = spec.parse().expect("STEP_CREASE_EDGE");
            eprintln!("  edge #{edge}:");
            for (index, face) in brep.faces.iter().enumerate() {
                let touches = face
                    .bounds
                    .iter()
                    .any(|bound| bound.half_edges.iter().any(|he| he.edge == edge));
                if touches {
                    let kind = match &face.surface {
                        crate::geometry::cad::brep::surface::Surface::Plane(p) => {
                            let o: [f64; 3] = std::array::from_fn(|k| p.origin[k].value());
                            let n: [f64; 3] = std::array::from_fn(|k| p.normal[k].value());
                            format!(
                                "plane origin=[{:.5} {:.5} {:.5}] normal=[{:.5} {:.5} {:.5}]",
                                o[0], o[1], o[2], n[0], n[1], n[2],
                            )
                        }
                        crate::geometry::cad::brep::surface::Surface::Cylinder(_) => {
                            "cylinder".into()
                        }
                        crate::geometry::cad::brep::surface::Surface::Cone(_) => "cone".into(),
                        crate::geometry::cad::brep::surface::Surface::Sphere(_) => "sphere".into(),
                        crate::geometry::cad::brep::surface::Surface::Torus(_) => "torus".into(),
                        crate::geometry::cad::brep::surface::Surface::BSpline(_) => {
                            "bspline".into()
                        }
                        crate::geometry::cad::brep::surface::Surface::Revolution(_) => {
                            "revolution".into()
                        }
                    };
                    eprintln!("    face #{index}: {kind}, forward={}", face.forward);
                    for (bi, bound) in face.bounds.iter().enumerate() {
                        eprint!("      bound {bi}:");
                        for he in &bound.half_edges {
                            eprint!(" e{}{}", he.edge, if he.forward { "+" } else { "-" });
                        }
                        eprintln!();
                    }
                }
            }
        }
    }
}

/// Whole-crease damage census: walks the straight line between the two
/// endpoint vertices of `STEP_CREASE_DAMAGE_EDGE` (a brep edge index, as
/// printed by `probe_creases`), and for every fitted hex whose centroid falls
/// within `STEP_CREASE_DAMAGE_BAND` (default 3e-4 m) of that line, reports its
/// max edge ratio, max skew, and min scaled Jacobian, binned by the point's
/// position along the line (`t` in 0..1). If the fit is dragging a whole band
/// of elements onto the crease uniformly along its length (not just at the two
/// ends), the edge-ratio/skew values will be elevated across the *entire*
/// `t` range, not just near `t=0` and `t=1` where the existing SJ<=0.1 census
/// already looks.
#[test]
#[ignore = "whole-crease elongation census for STEP_MESH_FILE / STEP_CREASE_DAMAGE_EDGE"]
fn probe_crease_damage() {
    use crate::{
        geometry::{
            cad::sizing::FeatureSizing,
            mesh::{Connectivity, Fitting, Verdict},
            ntree::Balancing,
            solid::Solid,
        },
        math::Quantity,
        units::Length,
    };
    let Ok(path) = std::env::var("STEP_MESH_FILE") else {
        return;
    };
    let Ok(edge_spec) = std::env::var("STEP_CREASE_DAMAGE_EDGE") else {
        return;
    };
    let edge: usize = edge_spec.parse().expect("STEP_CREASE_DAMAGE_EDGE");
    let env_f64 = |key, default: f64| -> f64 {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let length = |v| Quantity::<Length>::new(v);
    let text = std::fs::read_to_string(&path).unwrap();
    let breps = read_all(&text).expect("read failed");
    let levels = std::env::var("STEP_MESH_LEVELS")
        .ok()
        .and_then(|v| v.parse().ok());
    let cell = env_f64("STEP_MESH_CELL", 6.0e-3);
    let maximum =
        (std::env::var("STEP_MESH_CELL").as_deref() != Ok("none")).then(|| length(cell));
    let gradation = match std::env::var("STEP_MESH_GRADATION").as_deref() {
        Ok("none") => None,
        Ok(v) => Some(v.parse().expect("STEP_MESH_GRADATION")),
        Err(_) => Some(0.2),
    };
    let band = env_f64("STEP_CREASE_DAMAGE_BAND", 3.0e-4);
    // When set, discard any hex whose perpendicular foot falls off the segment
    // ends (or within this t-margin of an end). This purges the clamp
    // catch-all so buckets 0.00 and 0.95 contain only genuinely near-endpoint
    // hexes rather than the whole far mesh clamped to t=0/1.
    let interior_margin = std::env::var("STEP_CREASE_DAMAGE_INTERIOR")
        .ok()
        .and_then(|v| v.parse::<f64>().ok());

    for (index, brep) in breps.iter().enumerate() {
        eprintln!("--- solid {index}: {} faces ---", brep.faces.len());
        let [ia, ib] = brep.edges[edge].vertices;
        let a: [f64; 3] = std::array::from_fn(|k| brep.vertices[ia][k].value());
        let b: [f64; 3] = std::array::from_fn(|k| brep.vertices[ib][k].value());
        let d: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
        let len2 = d.iter().map(|v| v * v).sum::<f64>();
        eprintln!(
            "  edge #{edge}: a=[{:.5} {:.5} {:.5}] b=[{:.5} {:.5} {:.5}] len={:.6}",
            a[0], a[1], a[2], b[0], b[1], b[2], len2.sqrt(),
        );

        let mut sizing = FeatureSizing::of(
            brep,
            env_f64("STEP_MESH_SEGMENTS", 24.0) as usize,
            length(env_f64("STEP_MESH_MIN", cell / 8.0)),
            maximum,
            gradation,
        );
        if let Ok(n) = std::env::var("STEP_MESH_PROXIMITY") {
            sizing = sizing.with_proximity(brep, n.parse().unwrap()).unwrap();
        }
        if let Ok(n) = std::env::var("STEP_MESH_CURVATURE") {
            sizing = sizing.with_curvature(brep, n.parse().unwrap()).unwrap();
        }

        let mesh = brep
            .mesh(&sizing, levels, 0.1, Balancing::Strong(1), Fitting::Soft)
            .expect("mesh failed");
        let coords = mesh.coordinates();
        let ratios = mesh.maximum_edge_ratios();
        let skews = mesh.maximum_skews();
        let sj = mesh.minimum_scaled_jacobians();
        let [Connectivity::Hexahedral(block)] = mesh.connectivities() else {
            eprintln!("  not a single hex block");
            continue;
        };

        // Bin by t along the line into 20 buckets; track the worst (max)
        // edge-ratio/skew and worst (min) SJ per bucket, plus how many hexes
        // in-band land in that bucket.
        const BUCKETS: usize = 20;
        let mut bins = vec![(0usize, 0.0f64, 0.0f64, f64::INFINITY); BUCKETS];
        let mut in_band = 0usize;
        for (element, hex) in block.iter().enumerate() {
            let c: [f64; 3] = std::array::from_fn(|k| {
                hex.iter().map(|&n| coords[n][k].value()).sum::<f64>() / 8.0
            });
            // Project c onto the line, clamp to segment, measure perpendicular distance.
            let ac: [f64; 3] = std::array::from_fn(|k| c[k] - a[k]);
            let t_raw = ac.iter().zip(&d).map(|(x, y)| x * y).sum::<f64>() / len2;
            if let Some(m) = interior_margin {
                if t_raw < m || t_raw > 1.0 - m {
                    continue;
                }
            }
            let t = t_raw.clamp(0.0, 1.0);
            let proj: [f64; 3] = std::array::from_fn(|k| a[k] + t * d[k]);
            let dist = (0..3).map(|k| (c[k] - proj[k]).powi(2)).sum::<f64>().sqrt();
            if dist > band {
                continue;
            }
            in_band += 1;
            let bucket = ((t * BUCKETS as f64) as usize).min(BUCKETS - 1);
            let entry = &mut bins[bucket];
            entry.0 += 1;
            entry.1 = entry.1.max(ratios[0][element]);
            entry.2 = entry.2.max(skews[0][element]);
            entry.3 = entry.3.min(sj[0][element]);
        }
        eprintln!("  {in_band} hexes within {band:.1e} m of the crease line");
        eprintln!("  t-bucket  count  max-edge-ratio  max-skew  min-SJ");
        for (i, (n, ratio, skew, sjmin)) in bins.iter().enumerate() {
            if *n == 0 {
                eprintln!("    {:.2}       0      -               -         -", i as f64 / BUCKETS as f64);
                continue;
            }
            eprintln!(
                "    {:.2}    {n:5}      {ratio:8.3}      {skew:6.3}   {sjmin:+.4}",
                i as f64 / BUCKETS as f64,
            );
        }
    }
}

/// Walks straight along `STEP_CREASE_TIE_EDGE` (a brep edge index) and, at
/// each of `STEP_CREASE_TIE_SAMPLES` points offset `STEP_CREASE_TIE_OFFSET`
/// (default 5e-5 m) into the solid along the bisector of the two faces
/// adjacent to that edge, dumps `patch_report`'s top two nearest patches and
/// their normals. If the winning patch index alternates between the two
/// adjacent faces along the crease's length — rather than staying fixed or
/// changing only where a third face actually takes over — that is direct
/// evidence the fit's single-nearest-patch target is unstable along a sharp
/// concave edge (hypothesis b), which would explain uniformly elevated skew
/// down the whole crease even though the *position* target stays converged
/// (so SJ and interior-hole census don't flag it).
#[test]
#[ignore = "nearest-patch stability walk along a crease edge for STEP_MESH_FILE"]
fn probe_crease_tie() {
    use crate::geometry::Coordinate;
    let Ok(path) = std::env::var("STEP_MESH_FILE") else {
        return;
    };
    let Ok(edge_spec) = std::env::var("STEP_CREASE_TIE_EDGE") else {
        return;
    };
    let edge: usize = edge_spec.parse().expect("STEP_CREASE_TIE_EDGE");
    let samples: usize = std::env::var("STEP_CREASE_TIE_SAMPLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    let offset: f64 = std::env::var("STEP_CREASE_TIE_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5.0e-5);
    let text = std::fs::read_to_string(&path).unwrap();
    let breps = read_all(&text).expect("read failed");
    for (index, brep) in breps.iter().enumerate() {
        eprintln!("--- solid {index}: {} faces ---", brep.faces.len());
        let [ia, ib] = brep.edges[edge].vertices;
        let a: [f64; 3] = std::array::from_fn(|k| brep.vertices[ia][k].value());
        let b: [f64; 3] = std::array::from_fn(|k| brep.vertices[ib][k].value());
        let d: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);

        // Find the two faces touching this edge and grab their plane normals
        // to build an inward bisector offset.
        let mut normals: Vec<[f64; 3]> = Vec::new();
        for face in &brep.faces {
            let touches = face
                .bounds
                .iter()
                .any(|bound| bound.half_edges.iter().any(|he| he.edge == edge));
            if touches {
                if let crate::geometry::cad::brep::surface::Surface::Plane(p) = &face.surface {
                    let sign = if face.forward { 1.0 } else { -1.0 };
                    normals.push(std::array::from_fn(|k| sign * p.normal[k].value()));
                }
            }
        }
        if normals.len() != 2 {
            eprintln!("  edge #{edge}: expected 2 planar faces, found {}", normals.len());
            continue;
        }
        // Bisector pointing into the solid: average of the two outward
        // normals, negated (outward normals of a concave edge point apart;
        // their negated sum points into the material wedge between them).
        let mut bis: [f64; 3] = std::array::from_fn(|k| -(normals[0][k] + normals[1][k]));
        let bn = bis.iter().map(|v| v * v).sum::<f64>().sqrt();
        if bn > 1e-12 {
            for v in &mut bis {
                *v /= bn;
            }
        }
        eprintln!(
            "  face normals: {:?} {:?}  bisector (inward) {:?}",
            normals[0], normals[1], bis
        );

        let oracle = brep.oracle().expect("oracle");
        let mut last_patch: Option<usize> = None;
        let mut flips = 0usize;
        for i in 0..=samples {
            let t = i as f64 / samples as f64;
            let p: [f64; 3] = std::array::from_fn(|k| a[k] + t * d[k] + offset * bis[k]);
            let q = Coordinate::from(p);
            let report = oracle.patch_report(&q);
            let (i0, k0, d0, _, n0) = report[0];
            let (i1, k1, d1, _, n1) = report.get(1).copied().unwrap_or((usize::MAX, "-", f64::NAN, [0.0; 3], [0.0; 3]));
            if let Some(prev) = last_patch {
                if prev != i0 {
                    flips += 1;
                }
            }
            last_patch = Some(i0);
            eprintln!(
                "    t={t:.3} p=[{:.6} {:.6} {:.6}]  best #{i0} [{k0}] d={d0:.7} n=[{:.4} {:.4} {:.4}]  runner #{i1} [{k1}] d={d1:.7} n=[{:.4} {:.4} {:.4}]  gap={:.2e}",
                p[0], p[1], p[2], n0[0], n0[1], n0[2], n1[0], n1[1], n1[2], d1 - d0,
            );
        }
        eprintln!("  {flips} winning-patch flips over {samples} samples");
    }
}

#[test]
#[ignore = "dumps geometry + trim extent for one brep face, by index"]
fn probe_face_info() {
    let Ok(path) = std::env::var("STEP_MESH_FILE") else {
        return;
    };
    let Ok(spec) = std::env::var("STEP_FACE_INFO") else {
        return;
    };
    let index: usize = spec.parse().expect("STEP_FACE_INFO");
    let text = std::fs::read_to_string(&path).unwrap();
    let breps = read_all(&text).expect("read failed");
    for brep in &breps {
        let face = &brep.faces[index];
        eprintln!("face #{index}: forward={}", face.forward);
        match &face.surface {
            crate::geometry::cad::brep::surface::Surface::Cylinder(c) => {
                let o: [f64; 3] = std::array::from_fn(|k| c.origin[k].value());
                let a: [f64; 3] = std::array::from_fn(|k| c.axis[k].value());
                eprintln!("  cylinder origin=[{:.6} {:.6} {:.6}] axis=[{:.5} {:.5} {:.5}] radius={:.6}", o[0], o[1], o[2], a[0], a[1], a[2], c.radius);
            }
            crate::geometry::cad::brep::surface::Surface::Plane(p) => {
                let o: [f64; 3] = std::array::from_fn(|k| p.origin[k].value());
                let n: [f64; 3] = std::array::from_fn(|k| p.normal[k].value());
                eprintln!("  plane origin=[{:.6} {:.6} {:.6}] normal=[{:.5} {:.5} {:.5}]", o[0], o[1], o[2], n[0], n[1], n[2]);
            }
            _ => eprintln!("  other surface kind"),
        }
        for (bi, bound) in face.bounds.iter().enumerate() {
            eprint!("  bound {bi}:");
            for he in &bound.half_edges {
                let [ia, ib] = brep.edges[he.edge].vertices;
                let a: [f64; 3] = std::array::from_fn(|k| brep.vertices[ia][k].value());
                let b: [f64; 3] = std::array::from_fn(|k| brep.vertices[ib][k].value());
                eprint!(" e{}{}[{:.5},{:.5},{:.5} -> {:.5},{:.5},{:.5}]", he.edge, if he.forward {"+"} else {"-"}, a[0],a[1],a[2], b[0],b[1],b[2]);
            }
            eprintln!();
        }
    }
}

#[test]
#[ignore = "sizing field vs true nearby-surface gap along a crease's inward bisector"]
fn probe_crease_sizing() {
    use crate::{
        geometry::cad::sizing::FeatureSizing,
        math::Quantity,
        units::Length,
    };
    let Ok(path) = std::env::var("STEP_MESH_FILE") else {
        return;
    };
    let Ok(edge_spec) = std::env::var("STEP_CREASE_SIZING_EDGE") else {
        return;
    };
    let edge: usize = edge_spec.parse().expect("STEP_CREASE_SIZING_EDGE");
    let env_f64 = |key, default: f64| -> f64 {
        std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    };
    let length = |v| Quantity::<Length>::new(v);
    let text = std::fs::read_to_string(&path).unwrap();
    let breps = read_all(&text).expect("read failed");
    let cell = env_f64("STEP_MESH_CELL", 6.0e-3);
    let maximum = (std::env::var("STEP_MESH_CELL").as_deref() != Ok("none")).then(|| length(cell));
    let gradation = match std::env::var("STEP_MESH_GRADATION").as_deref() {
        Ok("none") => None,
        Ok(v) => Some(v.parse().expect("STEP_MESH_GRADATION")),
        Err(_) => Some(0.2),
    };
    for brep in &breps {
        let sizing = {
            let mut s = FeatureSizing::of(
                brep,
                env_f64("STEP_MESH_SEGMENTS", 24.0) as usize,
                length(env_f64("STEP_MESH_MIN", cell / 8.0)),
                maximum,
                gradation,
            );
            if let Ok(n) = std::env::var("STEP_MESH_PROXIMITY") {
                s = s.with_proximity(brep, n.parse().unwrap()).unwrap();
            }
            if let Ok(n) = std::env::var("STEP_MESH_CURVATURE") {
                s = s.with_curvature(brep, n.parse().unwrap()).unwrap();
            }
            s
        };
        let oracle = brep.oracle().expect("oracle");
        let [ia, ib] = brep.edges[edge].vertices;
        let a: [f64; 3] = std::array::from_fn(|k| brep.vertices[ia][k].value());
        let b: [f64; 3] = std::array::from_fn(|k| brep.vertices[ib][k].value());
        let d: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
        let mut normals: Vec<[f64; 3]> = Vec::new();
        for face in &brep.faces {
            let touches = face.bounds.iter().any(|bound| bound.half_edges.iter().any(|he| he.edge == edge));
            if touches {
                if let crate::geometry::cad::brep::surface::Surface::Plane(p) = &face.surface {
                    let sign = if face.forward { 1.0 } else { -1.0 };
                    normals.push(std::array::from_fn(|k| sign * p.normal[k].value()));
                }
            }
        }
        let mut bis: [f64; 3] = std::array::from_fn(|k| -(normals[0][k] + normals[1][k]));
        let bn = bis.iter().map(|v| v * v).sum::<f64>().sqrt();
        for v in &mut bis { *v /= bn; }
        for off in [1.0e-5, 2.0e-5, 3.0e-5, 5.0e-5, 8.0e-5, 1.5e-4] {
            let t = 0.5;
            let p: [f64; 3] = std::array::from_fn(|k| a[k] + t * d[k] + off * bis[k]);
            let q = crate::geometry::Coordinate::from(p);
            let target = sizing.at(&q).value();
            let report = oracle.patch_report(&q);
            let (i0, k0, d0, _, _) = report[0];
            eprintln!(
                "  offset {off:.1e}: sizing target = {target:.7}  nearest patch #{i0} [{k0}] d={d0:.7}",
            );
        }
    }
}

#[test]
#[ignore = "checks BrepOracle::local_diameter along a crease's inward bisector"]
fn probe_crease_local_diameter() {
    let Ok(path) = std::env::var("STEP_MESH_FILE") else {
        return;
    };
    let Ok(edge_spec) = std::env::var("STEP_CREASE_LD_EDGE") else {
        return;
    };
    let edge: usize = edge_spec.parse().expect("STEP_CREASE_LD_EDGE");
    let text = std::fs::read_to_string(&path).unwrap();
    let breps = read_all(&text).expect("read failed");
    for brep in &breps {
        let oracle = brep.oracle().expect("oracle");
        let [ia, ib] = brep.edges[edge].vertices;
        let a: [f64; 3] = std::array::from_fn(|k| brep.vertices[ia][k].value());
        let b: [f64; 3] = std::array::from_fn(|k| brep.vertices[ib][k].value());
        let d: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
        let mut normals: Vec<[f64; 3]> = Vec::new();
        for face in &brep.faces {
            let touches = face.bounds.iter().any(|bound| bound.half_edges.iter().any(|he| he.edge == edge));
            if touches {
                if let crate::geometry::cad::brep::surface::Surface::Plane(p) = &face.surface {
                    let sign = if face.forward { 1.0 } else { -1.0 };
                    normals.push(std::array::from_fn(|k| sign * p.normal[k].value()));
                }
            }
        }
        let mut bis: [f64; 3] = std::array::from_fn(|k| -(normals[0][k] + normals[1][k]));
        let bn = bis.iter().map(|v| v * v).sum::<f64>().sqrt();
        for v in &mut bis { *v /= bn; }
        for off in [1.0e-5, 2.0e-5, 3.0e-5, 5.0e-5] {
            for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let p: [f64; 3] = std::array::from_fn(|k| a[k] + t * d[k] + off * bis[k]);
                let q = crate::geometry::Coordinate::from(p);
                let ld = oracle.local_diameter(&q);
                eprintln!("  off={off:.1e} t={t:.2}: local_diameter = {ld:.7}");
            }
        }
    }
}

#[test]
#[ignore = "validates the ray_distance-along-bisector thickness measure for crease proximity"]
fn probe_crease_thickness_ray() {
    use crate::geometry::Coordinate;
    let Ok(path) = std::env::var("STEP_MESH_FILE") else {
        return;
    };
    let Ok(edge_spec) = std::env::var("STEP_CREASE_RAY_EDGE") else {
        return;
    };
    let edge: usize = edge_spec.parse().expect("STEP_CREASE_RAY_EDGE");
    let text = std::fs::read_to_string(&path).unwrap();
    let breps = read_all(&text).expect("read failed");
    for brep in &breps {
        let oracle = brep.oracle().expect("oracle");
        let [ia, ib] = brep.edges[edge].vertices;
        let a: [f64; 3] = std::array::from_fn(|k| brep.vertices[ia][k].value());
        let b: [f64; 3] = std::array::from_fn(|k| brep.vertices[ib][k].value());
        let d: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
        let mut normals: Vec<[f64; 3]> = Vec::new();
        for face in &brep.faces {
            let touches = face.bounds.iter().any(|bound| bound.half_edges.iter().any(|he| he.edge == edge));
            if touches {
                if let crate::geometry::cad::brep::surface::Surface::Plane(p) = &face.surface {
                    let sign = if face.forward { 1.0 } else { -1.0 };
                    normals.push(std::array::from_fn(|k| sign * p.normal[k].value()));
                }
            }
        }
        let mut bis: [f64; 3] = std::array::from_fn(|k| -(normals[0][k] + normals[1][k]));
        let bn = bis.iter().map(|v| v * v).sum::<f64>().sqrt();
        for v in &mut bis { *v /= bn; }
        let eps = 1.0e-6;
        for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            let base: [f64; 3] = std::array::from_fn(|k| a[k] + t * d[k]);
            let origin: [f64; 3] = std::array::from_fn(|k| base[k] + eps * bis[k]);
            let hit = oracle.ray_distance(&Coordinate::from(origin), bis);
            let thickness = hit.map(|h| h + eps);
            eprintln!("  t={t:.2}: ray-along-bisector thickness = {thickness:?}");
        }
    }
}

/// Verifies the medial-axis-ambiguity hypothesis directly on the real
/// trimmed-mesh boundary quads near a crease, rather than on a synthetic
/// bisector line: builds the actual `trim()` mesh (no fitting), finds every
/// exterior boundary quad whose centroid lies within `STEP_CREASE_QUAD_BAND`
/// of the crease edge's segment, and for each reports the top-2
/// `patch_report` candidates plus which one is closer — so a real flip
/// between neighboring quads (not just a hypothetical offset line) is either
/// confirmed or ruled out.
#[test]
#[ignore = "checks whether real trimmed-mesh boundary quads near a crease flip target patch"]
fn probe_crease_quad_flips() {
    use crate::{
        geometry::{
            cad::sizing::FeatureSizing,
            ntree::Balancing,
            solid::Solid,
        },
        math::Quantity,
        units::Length,
    };
    let Ok(path) = std::env::var("STEP_MESH_FILE") else {
        return;
    };
    let Ok(edge_spec) = std::env::var("STEP_CREASE_QUAD_EDGE") else {
        return;
    };
    let edge: usize = edge_spec.parse().expect("STEP_CREASE_QUAD_EDGE");
    let env_f64 = |key, default: f64| -> f64 {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let length = |v| Quantity::<Length>::new(v);
    let text = std::fs::read_to_string(&path).unwrap();
    let breps = read_all(&text).expect("read failed");
    let levels = std::env::var("STEP_MESH_LEVELS")
        .ok()
        .and_then(|v| v.parse().ok());
    let cell = env_f64("STEP_MESH_CELL", 6.0e-3);
    let maximum =
        (std::env::var("STEP_MESH_CELL").as_deref() != Ok("none")).then(|| length(cell));
    let gradation = match std::env::var("STEP_MESH_GRADATION").as_deref() {
        Ok("none") => None,
        Ok(v) => Some(v.parse().expect("STEP_MESH_GRADATION")),
        Err(_) => Some(0.2),
    };
    let band = env_f64("STEP_CREASE_QUAD_BAND", 3.0e-4);

    for (index, brep) in breps.iter().enumerate() {
        eprintln!("--- solid {index}: {} faces ---", brep.faces.len());
        let [ia, ib] = brep.edges[edge].vertices;
        let a: [f64; 3] = std::array::from_fn(|k| brep.vertices[ia][k].value());
        let b: [f64; 3] = std::array::from_fn(|k| brep.vertices[ib][k].value());
        let d: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
        let len2 = d.iter().map(|v| v * v).sum::<f64>();

        let mut sizing = FeatureSizing::of(
            brep,
            env_f64("STEP_MESH_SEGMENTS", 24.0) as usize,
            length(env_f64("STEP_MESH_MIN", cell / 8.0)),
            maximum,
            gradation,
        );
        if let Ok(n) = std::env::var("STEP_MESH_PROXIMITY") {
            sizing = sizing.with_proximity(brep, n.parse().unwrap()).unwrap();
        }
        if let Ok(n) = std::env::var("STEP_MESH_CURVATURE") {
            sizing = sizing.with_curvature(brep, n.parse().unwrap()).unwrap();
        }

        let (trimmed, _) = brep
            .trim(&sizing, levels, 0.1, Balancing::Strong(1))
            .expect("trim failed");
        let oracle = brep.oracle().expect("oracle");
        let coords = trimmed.coordinates();
        let quads = trimmed.exterior_faces();

        // Bin surviving quads by t along the crease's segment (clamp to
        // [0,1]) so output reads in order along the crease's length. Each row
        // carries both the plain-nearest winner (i0) and the visibility-gated
        // fit target (ig), so the flip reduction is measured directly.
        let mut rows: Vec<(f64, [f64; 3], usize, &'static str, f64, usize, &'static str, f64, usize)> =
            Vec::new();
        for quad in &quads {
            if quad.len() != 4 {
                continue;
            }
            let centroid: [f64; 3] = std::array::from_fn(|k| {
                quad.iter().map(|&n| coords[n][k].value()).sum::<f64>() / 4.0
            });
            let ac: [f64; 3] = std::array::from_fn(|k| centroid[k] - a[k]);
            let t = ac.iter().zip(&d).map(|(x, y)| x * y).sum::<f64>() / len2;
            let tc = t.clamp(0.0, 1.0);
            let closest: [f64; 3] = std::array::from_fn(|k| a[k] + tc * d[k]);
            let perp = centroid
                .iter()
                .zip(&closest)
                .map(|(c, l)| (c - l).powi(2))
                .sum::<f64>()
                .sqrt();
            if perp > band {
                continue;
            }
            let q = crate::geometry::Coordinate::from(centroid);
            let report = oracle.patch_report(&q);
            let (i0, k0, d0, _, _) = report[0];
            let (i1, k1, d1, _, _) = report.get(1).copied().unwrap_or((usize::MAX, "-", f64::NAN, [0.0; 3], [0.0; 3]));
            let ig = oracle.fit_target(&q).map_or(usize::MAX, |(i, _)| i);
            rows.push((t, centroid, i0, k0, d0, i1, k1, d1 - d0, ig));
        }
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        eprintln!("  {} boundary quads within {band:.1e} m of crease #{edge}", rows.len());
        let mut last: Option<usize> = None;
        let mut last_gated: Option<usize> = None;
        let mut flips = 0usize;
        let mut gated_flips = 0usize;
        for (t, c, i0, k0, d0, i1, k1, gap, ig) in &rows {
            if let Some(prev) = last {
                if prev != *i0 {
                    flips += 1;
                }
            }
            if let Some(prev) = last_gated {
                if prev != *ig {
                    gated_flips += 1;
                }
            }
            last = Some(*i0);
            last_gated = Some(*ig);
            eprintln!(
                "    t={t:6.3} c=[{:.6} {:.6} {:.6}]  best #{i0} [{k0}] d={d0:.7}  runner #{i1} [{k1}] gap={gap:.2e}  gated #{ig}",
                c[0], c[1], c[2],
            );
        }
        eprintln!(
            "  {flips} plain-nearest flips, {gated_flips} gated flips over {} quads",
            rows.len()
        );
    }
}
