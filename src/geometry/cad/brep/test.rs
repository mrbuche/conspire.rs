use super::{
    Brep, Edge, Face, HalfEdge, Loop, Shell,
    curve::{Curve, Line},
    surface::{Plane, Surface},
};
use crate::geometry::{Coordinate, Direction};

pub(crate) fn direction(entries: [f64; 3]) -> Direction<3> {
    Direction::const_from(entries)
}

pub(crate) fn edge(a: usize, b: usize) -> Edge {
    Edge {
        vertices: [a, b],
        curve: Curve::Line(Line {
            origin: Coordinate::const_from([0.0; 3]),
            direction: direction([1.0, 0.0, 0.0]),
        }),
    }
}

/// `half_edges[i]` is `(edge index, forward?)`.
pub(crate) fn face(normal: [f64; 3], reference: [f64; 3], half_edges: &[(usize, bool)]) -> Face {
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

/// The axis-aligned unit cube `[0, 1]^3` as a closed shell of six planar faces.
pub(crate) fn unit_cube() -> Brep {
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

/// Two unit squares in the `z = 0` plane sharing edge `1` (`v1`-`v4`). An open
/// shell: the six perimeter edges are boundaries and the shared edge is flat.
///
/// ```text
///   v5 --- v4 --- v3
///   |  L   |  R   |
///   v0 --- v1 --- v2
/// ```
pub(crate) fn coplanar_squares() -> Brep {
    let vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    .into_iter()
    .map(Coordinate::const_from)
    .collect();
    let edges = vec![
        edge(0, 1), // 0
        edge(1, 4), // 1  shared
        edge(4, 5), // 2
        edge(5, 0), // 3
        edge(1, 2), // 4
        edge(2, 3), // 5
        edge(3, 4), // 6
    ];
    let up = [0.0, 0.0, 1.0];
    let reference = [1.0, 0.0, 0.0];
    let faces = vec![
        face(up, reference, &[(0, true), (1, true), (2, true), (3, true)]),
        face(
            up,
            reference,
            &[(4, true), (5, true), (6, true), (1, false)],
        ),
    ];
    Brep {
        vertices,
        edges,
        faces,
        shells: vec![Shell {
            faces: vec![0, 1],
            closed: false,
        }],
    }
}
