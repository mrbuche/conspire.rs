use super::{
    Brep, Edge, Face, HalfEdge, Loop, Shell,
    curve::{Circle, Curve, Ellipse, Line},
    surface::{Cone, Cylinder, Plane, Sphere, Surface, Torus},
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
        poles: vec![],
        forward: true,
    }
}

/// The axis-aligned unit cube `[0, 1]^3` as a closed shell of six planar faces.
pub(crate) fn unit_cube() -> Brep {
    axis_aligned_box([1.0, 1.0, 1.0])
}

/// The axis-aligned box `[0, extents[0]] x [0, extents[1]] x [0, extents[2]]` as a
/// closed shell of six planar faces.
pub(crate) fn axis_aligned_box(extents: [f64; 3]) -> Brep {
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
    .map(|[x, y, z]| Coordinate::const_from([x * extents[0], y * extents[1], z * extents[2]]))
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

/// A capped right circular cylinder about `+z`, base centred at the origin: two
/// planar disk caps and one cylindrical lateral face split by a seam line at
/// angle 0.
pub(crate) fn capped_cylinder(radius: f64, height: f64) -> Brep {
    let vertices = vec![
        Coordinate::const_from([radius, 0.0, 0.0]),
        Coordinate::const_from([radius, 0.0, height]),
    ];
    let rim = |z: f64| {
        Curve::Circle(Circle {
            center: Coordinate::const_from([0.0, 0.0, z]),
            axis: direction([0.0, 0.0, 1.0]),
            reference_direction: direction([1.0, 0.0, 0.0]),
            radius,
        })
    };
    let edges = vec![
        Edge {
            vertices: [0, 0],
            curve: rim(0.0),
        },
        Edge {
            vertices: [1, 1],
            curve: rim(height),
        },
        Edge {
            vertices: [0, 1],
            curve: Curve::Line(Line {
                origin: Coordinate::const_from([radius, 0.0, 0.0]),
                direction: direction([0.0, 0.0, 1.0]),
            }),
        },
    ];
    let edge_loop = |half_edges: &[(usize, bool)]| Loop {
        half_edges: half_edges
            .iter()
            .map(|&(edge, forward)| HalfEdge { edge, forward })
            .collect(),
    };
    let faces = vec![
        Face {
            surface: Surface::Plane(Plane {
                origin: Coordinate::const_from([0.0, 0.0, 0.0]),
                normal: direction([0.0, 0.0, -1.0]),
                reference_direction: direction([1.0, 0.0, 0.0]),
            }),
            bounds: vec![edge_loop(&[(0, false)])],
            poles: vec![],
            forward: true,
        },
        Face {
            surface: Surface::Plane(Plane {
                origin: Coordinate::const_from([0.0, 0.0, height]),
                normal: direction([0.0, 0.0, 1.0]),
                reference_direction: direction([1.0, 0.0, 0.0]),
            }),
            bounds: vec![edge_loop(&[(1, true)])],
            poles: vec![],
            forward: true,
        },
        Face {
            surface: Surface::Cylinder(Cylinder {
                origin: Coordinate::const_from([0.0, 0.0, 0.0]),
                axis: direction([0.0, 0.0, 1.0]),
                reference_direction: direction([1.0, 0.0, 0.0]),
                radius,
            }),
            bounds: vec![edge_loop(&[(0, true), (2, true), (1, false), (2, false)])],
            poles: vec![],
            forward: true,
        },
    ];
    Brep {
        vertices,
        edges,
        faces,
        shells: vec![Shell {
            faces: vec![0, 1, 2],
            closed: true,
        }],
    }
}

/// A single cylindrical face about `+z`, radius `radius`: a genuine partial
/// sweep from angle `0` to `angle` (a fillet/chamfer remnant), not a full
/// turn — two rulings and two arc rims bounding one lateral patch.
pub(crate) fn partial_cylinder(radius: f64, height: f64, angle: f64) -> Brep {
    let point = |a: f64, z: f64| [radius * a.cos(), radius * a.sin(), z];
    let vertices = [point(0.0, 0.0), point(angle, 0.0), point(angle, height), point(0.0, height)]
        .map(Coordinate::const_from)
        .to_vec();
    let edges = vec![
        Edge {
            vertices: [0, 1],
            curve: Curve::Circle(Circle {
                center: Coordinate::const_from([0.0, 0.0, 0.0]),
                axis: direction([0.0, 0.0, 1.0]),
                reference_direction: direction([1.0, 0.0, 0.0]),
                radius,
            }),
        },
        Edge {
            vertices: [1, 2],
            curve: Curve::Line(Line {
                origin: Coordinate::const_from(point(angle, 0.0)),
                direction: direction([0.0, 0.0, 1.0]),
            }),
        },
        Edge {
            vertices: [2, 3],
            curve: Curve::Circle(Circle {
                center: Coordinate::const_from([0.0, 0.0, height]),
                axis: direction([0.0, 0.0, -1.0]),
                reference_direction: direction([1.0, 0.0, 0.0]),
                radius,
            }),
        },
        Edge {
            vertices: [3, 0],
            curve: Curve::Line(Line {
                origin: Coordinate::const_from(point(0.0, height)),
                direction: direction([0.0, 0.0, -1.0]),
            }),
        },
    ];
    let faces = vec![Face {
        surface: Surface::Cylinder(Cylinder {
            origin: Coordinate::const_from([0.0, 0.0, 0.0]),
            axis: direction([0.0, 0.0, 1.0]),
            reference_direction: direction([1.0, 0.0, 0.0]),
            radius,
        }),
        bounds: vec![Loop {
            half_edges: [(0, true), (1, true), (2, true), (3, true)]
                .into_iter()
                .map(|(edge, forward)| HalfEdge { edge, forward })
                .collect(),
        }],
        poles: vec![],
        forward: true,
    }];
    Brep {
        vertices,
        edges,
        faces,
        shells: vec![Shell { faces: vec![0], closed: false }],
    }
}

/// A single `+z` planar face whose outer loop is three straight sides plus a
/// semicircular arc bulging to `y = 6` — well past every vertex (max vertex
/// `y = 4`). The arc is on the *outer* bound, so the loop-vertex AABB alone
/// misses the bulge.
pub(crate) fn bulged_plate() -> Brep {
    let vertices = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 4.0, 0.0], [0.0, 4.0, 0.0]]
        .map(Coordinate::const_from)
        .to_vec();
    let line = |a: usize, b: usize| Edge {
        vertices: [a, b],
        curve: Curve::Line(Line {
            origin: vertices[a].clone(),
            direction: direction([1.0, 0.0, 0.0]),
        }),
    };
    let edges = vec![
        line(0, 1),
        line(1, 2),
        Edge {
            vertices: [2, 3],
            curve: Curve::Circle(Circle {
                center: Coordinate::const_from([2.0, 4.0, 0.0]),
                axis: direction([0.0, 0.0, 1.0]),
                reference_direction: direction([1.0, 0.0, 0.0]),
                radius: 2.0,
            }),
        },
        line(3, 0),
    ];
    let faces = vec![Face {
        surface: Surface::Plane(Plane {
            origin: Coordinate::const_from([0.0, 0.0, 0.0]),
            normal: direction([0.0, 0.0, 1.0]),
            reference_direction: direction([1.0, 0.0, 0.0]),
        }),
        bounds: vec![Loop {
            half_edges: [(0, true), (1, true), (2, true), (3, true)]
                .into_iter()
                .map(|(edge, forward)| HalfEdge { edge, forward })
                .collect(),
        }],
        poles: vec![],
        forward: true,
    }];
    Brep {
        vertices,
        edges,
        faces,
        shells: vec![Shell { faces: vec![0], closed: false }],
    }
}

/// A spherical face bounded by a single equator circle traversed once — a
/// genuine partial (hemisphere) patch, not a whole sphere closed by a seam.
pub(crate) fn partial_sphere(radius: f64) -> Brep {
    let vertices = vec![Coordinate::const_from([radius, 0.0, 0.0])];
    let edges = vec![Edge {
        vertices: [0, 0],
        curve: Curve::Circle(Circle {
            center: Coordinate::const_from([0.0, 0.0, 0.0]),
            axis: direction([0.0, 0.0, 1.0]),
            reference_direction: direction([1.0, 0.0, 0.0]),
            radius,
        }),
    }];
    let faces = vec![Face {
        surface: Surface::Sphere(Sphere {
            origin: Coordinate::const_from([0.0, 0.0, 0.0]),
            axis: direction([0.0, 0.0, 1.0]),
            reference_direction: direction([1.0, 0.0, 0.0]),
            radius,
        }),
        bounds: vec![Loop {
            half_edges: vec![HalfEdge { edge: 0, forward: true }],
        }],
        poles: vec![],
        forward: true,
    }];
    Brep {
        vertices,
        edges,
        faces,
        shells: vec![Shell { faces: vec![0], closed: false }],
    }
}

/// A truncated cone about `+z`: `base_radius` at `z = 0`, `tip_radius` at
/// `z = height`, one conical lateral face split by a seam line.
pub(crate) fn cone(base_radius: f64, tip_radius: f64, height: f64) -> Brep {
    let widening = tip_radius >= base_radius;
    let (origin_z, cone_axis, cone_radius, delta) = if widening {
        (0.0, [0.0, 0.0, 1.0], base_radius, tip_radius - base_radius)
    } else {
        (height, [0.0, 0.0, -1.0], tip_radius, base_radius - tip_radius)
    };
    let semi_angle = (delta / height).atan();

    let vertices = vec![
        Coordinate::const_from([base_radius, 0.0, 0.0]),
        Coordinate::const_from([tip_radius, 0.0, height]),
    ];
    let rim = |z: f64, radius: f64| {
        Curve::Circle(Circle {
            center: Coordinate::const_from([0.0, 0.0, z]),
            axis: direction([0.0, 0.0, 1.0]),
            reference_direction: direction([1.0, 0.0, 0.0]),
            radius,
        })
    };
    let slant = {
        let d = [tip_radius - base_radius, 0.0, height];
        let n = (d[0] * d[0] + d[2] * d[2]).sqrt();
        direction([d[0] / n, 0.0, d[2] / n])
    };
    let edges = vec![
        Edge {
            vertices: [0, 0],
            curve: rim(0.0, base_radius),
        },
        Edge {
            vertices: [1, 1],
            curve: rim(height, tip_radius),
        },
        Edge {
            vertices: [0, 1],
            curve: Curve::Line(Line {
                origin: Coordinate::const_from([base_radius, 0.0, 0.0]),
                direction: slant,
            }),
        },
    ];
    let edge_loop = |half_edges: &[(usize, bool)]| Loop {
        half_edges: half_edges
            .iter()
            .map(|&(edge, forward)| HalfEdge { edge, forward })
            .collect(),
    };
    let faces = vec![
        Face {
            surface: Surface::Plane(Plane {
                origin: Coordinate::const_from([0.0, 0.0, 0.0]),
                normal: direction([0.0, 0.0, -1.0]),
                reference_direction: direction([1.0, 0.0, 0.0]),
            }),
            bounds: vec![edge_loop(&[(0, false)])],
            poles: vec![],
            forward: true,
        },
        Face {
            surface: Surface::Plane(Plane {
                origin: Coordinate::const_from([0.0, 0.0, height]),
                normal: direction([0.0, 0.0, 1.0]),
                reference_direction: direction([1.0, 0.0, 0.0]),
            }),
            bounds: vec![edge_loop(&[(1, true)])],
            poles: vec![],
            forward: true,
        },
        Face {
            surface: Surface::Cone(Cone {
                origin: Coordinate::const_from([0.0, 0.0, origin_z]),
                axis: direction(cone_axis),
                reference_direction: direction([1.0, 0.0, 0.0]),
                radius: cone_radius,
                semi_angle,
            }),
            bounds: vec![edge_loop(&[(0, true), (2, true), (1, false), (2, false)])],
            poles: vec![],
            forward: true,
        },
    ];
    Brep {
        vertices,
        edges,
        faces,
        shells: vec![Shell {
            faces: vec![0, 1, 2],
            closed: true,
        }],
    }
}

/// A ring torus about `+z` centred at the origin: one toroidal face closed by a
/// meridian seam and a longitude seam through the outer-equator point.
pub(crate) fn torus(major_radius: f64, minor_radius: f64) -> Brep {
    let seam = major_radius + minor_radius;
    let vertices = vec![Coordinate::const_from([seam, 0.0, 0.0])];
    let edges = vec![
        // Tube cross-section circle in the x-z plane at angle 0.
        Edge {
            vertices: [0, 0],
            curve: Curve::Circle(Circle {
                center: Coordinate::const_from([major_radius, 0.0, 0.0]),
                axis: direction([0.0, 1.0, 0.0]),
                reference_direction: direction([1.0, 0.0, 0.0]),
                radius: minor_radius,
            }),
        },
        // Outer equator traced by the seam point around the axis.
        Edge {
            vertices: [0, 0],
            curve: Curve::Circle(Circle {
                center: Coordinate::const_from([0.0, 0.0, 0.0]),
                axis: direction([0.0, 0.0, 1.0]),
                reference_direction: direction([1.0, 0.0, 0.0]),
                radius: seam,
            }),
        },
    ];
    let faces = vec![Face {
        surface: Surface::Torus(Torus {
            origin: Coordinate::const_from([0.0, 0.0, 0.0]),
            axis: direction([0.0, 0.0, 1.0]),
            reference_direction: direction([1.0, 0.0, 0.0]),
            major_radius,
            minor_radius,
        }),
        bounds: vec![Loop {
            half_edges: vec![
                HalfEdge {
                    edge: 0,
                    forward: true,
                },
                HalfEdge {
                    edge: 1,
                    forward: true,
                },
                HalfEdge {
                    edge: 0,
                    forward: false,
                },
                HalfEdge {
                    edge: 1,
                    forward: false,
                },
            ],
        }],
        poles: vec![],
        forward: true,
    }];
    Brep {
        vertices,
        edges,
        faces,
        shells: vec![Shell {
            faces: vec![0],
            closed: true,
        }],
    }
}

/// A sphere of `radius` centred at the origin.
pub(crate) fn ball(radius: f64) -> Brep {
    ball_at([0.0, 0.0, 0.0], radius)
}

/// A sphere of `radius` centred at `center`: one periodic spherical face with a
/// meridian seam between the two pole vertices.
pub(crate) fn ball_at(center: [f64; 3], radius: f64) -> Brep {
    let at = |offset: [f64; 3]| {
        Coordinate::const_from([
            center[0] + offset[0],
            center[1] + offset[1],
            center[2] + offset[2],
        ])
    };
    let vertices = vec![at([0.0, 0.0, -radius]), at([0.0, 0.0, radius])];
    let edges = vec![Edge {
        vertices: [0, 1],
        curve: Curve::Circle(Circle {
            center: at([0.0, 0.0, 0.0]),
            axis: direction([0.0, 1.0, 0.0]),
            reference_direction: direction([0.0, 0.0, -1.0]),
            radius,
        }),
    }];
    let faces = vec![Face {
        surface: Surface::Sphere(Sphere {
            origin: at([0.0, 0.0, 0.0]),
            axis: direction([0.0, 0.0, 1.0]),
            reference_direction: direction([1.0, 0.0, 0.0]),
            radius,
        }),
        bounds: vec![Loop {
            half_edges: vec![
                HalfEdge {
                    edge: 0,
                    forward: true,
                },
                HalfEdge {
                    edge: 0,
                    forward: false,
                },
            ],
        }],
        poles: vec![],
        forward: true,
    }];
    Brep {
        vertices,
        edges,
        faces,
        shells: vec![Shell {
            faces: vec![0],
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

/// A single `10x10` planar face (`+z` normal) with a rounded-rectangle hole
/// (corner radius `1`, straight sides in between) — a mix of straight and
/// circular-arc edges in one bound, like a real fillet-cornered pocket.
pub(crate) fn square_with_rounded_hole() -> Brep {
    let outer = [
        Coordinate::const_from([0.0, 0.0, 0.0]),
        Coordinate::const_from([10.0, 0.0, 0.0]),
        Coordinate::const_from([10.0, 10.0, 0.0]),
        Coordinate::const_from([0.0, 10.0, 0.0]),
    ];
    let hole = [
        Coordinate::const_from([3.0, 2.0, 0.0]),
        Coordinate::const_from([7.0, 2.0, 0.0]),
        Coordinate::const_from([8.0, 3.0, 0.0]),
        Coordinate::const_from([8.0, 7.0, 0.0]),
        Coordinate::const_from([7.0, 8.0, 0.0]),
        Coordinate::const_from([3.0, 8.0, 0.0]),
        Coordinate::const_from([2.0, 7.0, 0.0]),
        Coordinate::const_from([2.0, 3.0, 0.0]),
    ];
    let corners = [[7.0, 3.0], [7.0, 7.0], [3.0, 7.0], [3.0, 3.0]];
    let vertices: Vec<Coordinate<3>> = outer.into_iter().chain(hole).collect();
    let line = |a: usize, b: usize| Edge {
        vertices: [a, b],
        curve: Curve::Line(Line {
            origin: vertices[a].clone(),
            direction: direction([1.0, 0.0, 0.0]),
        }),
    };
    let arc = |a: usize, b: usize, [cx, cy]: [f64; 2]| Edge {
        vertices: [a, b],
        curve: Curve::Circle(Circle {
            center: Coordinate::const_from([cx, cy, 0.0]),
            axis: direction([0.0, 0.0, 1.0]),
            reference_direction: direction([1.0, 0.0, 0.0]),
            radius: 1.0,
        }),
    };
    let edges = vec![
        line(0, 1),
        line(1, 2),
        line(2, 3),
        line(3, 0),
        line(4, 5),
        arc(5, 6, corners[0]),
        line(6, 7),
        arc(7, 8, corners[1]),
        line(8, 9),
        arc(9, 10, corners[2]),
        line(10, 11),
        arc(11, 4, corners[3]),
    ];
    let edge_loop = |half_edges: &[usize]| Loop {
        half_edges: half_edges
            .iter()
            .map(|&edge| HalfEdge { edge, forward: true })
            .collect(),
    };
    let faces = vec![Face {
        surface: Surface::Plane(Plane {
            origin: Coordinate::const_from([0.0, 0.0, 0.0]),
            normal: direction([0.0, 0.0, 1.0]),
            reference_direction: direction([1.0, 0.0, 0.0]),
        }),
        bounds: vec![
            edge_loop(&[0, 1, 2, 3]),
            edge_loop(&[4, 5, 6, 7, 8, 9, 10, 11]),
        ],
        poles: vec![],
        forward: true,
    }];
    Brep {
        vertices,
        edges,
        faces,
        shells: vec![Shell { faces: vec![0], closed: false }],
    }
}

/// A single cylindrical face about `+z`, radius `radius`: a genuine partial
/// sweep from angle `0` to `angle`, flat-circle rim at the bottom and a
/// genuinely tilted elliptical rim (an oblique planar cut) at the top — the
/// `[Line, Circle, Line, Ellipse]` shape seen on real chamfered parts.
pub(crate) fn cylinder_with_elliptical_rim(radius: f64, angle: f64) -> Brep {
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let top = |a: f64| 5.0 - radius * a.sin();
    let point = |a: f64, z: f64| [radius * a.cos(), radius * a.sin(), z];
    let vertices = [
        point(0.0, 0.0),
        point(angle, 0.0),
        point(angle, top(angle)),
        point(0.0, top(0.0)),
    ]
    .map(Coordinate::const_from)
    .to_vec();
    let edges = vec![
        Edge {
            vertices: [0, 1],
            curve: Curve::Circle(Circle {
                center: Coordinate::const_from([0.0, 0.0, 0.0]),
                axis: direction([0.0, 0.0, 1.0]),
                reference_direction: direction([1.0, 0.0, 0.0]),
                radius,
            }),
        },
        Edge {
            vertices: [1, 2],
            curve: Curve::Line(Line {
                origin: Coordinate::const_from(point(angle, 0.0)),
                direction: direction([0.0, 0.0, 1.0]),
            }),
        },
        Edge {
            vertices: [2, 3],
            curve: Curve::Ellipse(Ellipse {
                center: Coordinate::const_from([0.0, 0.0, 5.0]),
                axis: direction([0.0, h, h]),
                reference_direction: direction([1.0, 0.0, 0.0]),
                major_radius: radius * std::f64::consts::SQRT_2,
                minor_radius: radius,
            }),
        },
        Edge {
            vertices: [3, 0],
            curve: Curve::Line(Line {
                origin: Coordinate::const_from(point(0.0, top(0.0))),
                direction: direction([0.0, 0.0, -1.0]),
            }),
        },
    ];
    let faces = vec![Face {
        surface: Surface::Cylinder(Cylinder {
            origin: Coordinate::const_from([0.0, 0.0, 0.0]),
            axis: direction([0.0, 0.0, 1.0]),
            reference_direction: direction([1.0, 0.0, 0.0]),
            radius,
        }),
        bounds: vec![Loop {
            half_edges: [(0, true), (1, true), (2, true), (3, true)]
                .into_iter()
                .map(|(edge, forward)| HalfEdge { edge, forward })
                .collect(),
        }],
        poles: vec![],
        forward: true,
    }];
    Brep {
        vertices,
        edges,
        faces,
        shells: vec![Shell { faces: vec![0], closed: false }],
    }
}
