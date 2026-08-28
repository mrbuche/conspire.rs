use crate::geometry::{
    Coordinate,
    cad::brep::{
        Brep, Edge, Face, HalfEdge, Loop, Shell,
        curve::{Curve, Line},
        surface::{Plane, Surface},
    },
    mesh::Connectivity,
};

fn direction(entries: [f64; 3]) -> crate::geometry::Direction<3> {
    crate::geometry::Direction::const_from(entries)
}

fn edge(a: usize, b: usize) -> Edge {
    Edge {
        vertices: [a, b],
        curve: Curve::Line(Line {
            origin: Coordinate::const_from([0.0; 3]),
            direction: direction([1.0, 0.0, 0.0]),
        }),
    }
}

/// `half_edges[i]` is `(edge index, forward?)`.
fn face(normal: [f64; 3], reference: [f64; 3], half_edges: &[(usize, bool)]) -> Face {
    Face {
        surface: Surface::Plane(Plane {
            origin: Coordinate::const_from([0.0; 3]),
            normal: direction(normal),
            reference_direction: direction(reference),
        }),
        bounds: vec![Loop {
            half_edges: half_edges
                .iter()
                .map(|&(edge, forward)| HalfEdge { edge, forward })
                .collect(),
        }],
        forward: true,
    }
}

fn unit_cube() -> Brep {
    let vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    .into_iter()
    .map(Coordinate::const_from)
    .collect();
    let edges = vec![
        edge(0, 1),
        edge(1, 2),
        edge(2, 3),
        edge(3, 0),
        edge(4, 5),
        edge(5, 6),
        edge(6, 7),
        edge(7, 4),
        edge(0, 4),
        edge(1, 5),
        edge(2, 6),
        edge(3, 7),
    ];
    let faces = vec![
        face(
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            &[(3, false), (2, false), (1, false), (0, false)],
        ),
        face(
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            &[(4, true), (5, true), (6, true), (7, true)],
        ),
        face(
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            &[(0, true), (9, true), (4, false), (8, false)],
        ),
        face(
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            &[(11, true), (6, false), (10, false), (2, true)],
        ),
        face(
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            &[(8, true), (7, false), (11, false), (3, true)],
        ),
        face(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            &[(1, true), (10, true), (5, false), (9, false)],
        ),
    ];
    Brep {
        vertices,
        edges,
        faces,
        shells: vec![Shell {
            faces: (0..6).collect(),
            closed: true,
        }],
    }
}

#[test]
fn tessellates_unit_cube() {
    let tessellation = unit_cube().tessellate().unwrap();
    let mesh = tessellation.mesh();
    assert_eq!(mesh.number_of_nodes(), 8);
    let Connectivity::Triangular(block) = &mesh.connectivities()[0] else {
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

#[test]
fn rejects_faces_with_holes() {
    let mut brep = unit_cube();
    brep.faces[0].bounds.push(Loop { half_edges: vec![] });
    assert_eq!(
        brep.tessellate().err(),
        Some("faces with holes are not yet supported")
    );
}
