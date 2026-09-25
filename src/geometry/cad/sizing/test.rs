use super::{FeatureSizing, shares_a_vertex};
use crate::{
    geometry::{
        Coordinate,
        cad::brep::{
            Brep, Shell,
            test::{edge, face, unit_cube},
        },
    },
    math::Quantity,
    units::Length,
};

fn length(value: f64) -> Quantity<Length> {
    Quantity::new(value)
}

fn point(coordinates: [f64; 3]) -> Coordinate<3> {
    Coordinate::const_from(coordinates)
}

#[test]
fn grows_from_the_edges_inward() {
    let field = FeatureSizing::of(
        &unit_cube(),
        2,
        Some(length(0.05)),
        Some(length(10.0)),
        Some(1.0),
    );
    let on_edge = field.at(&point([0.5, 0.0, 0.0])).value();
    let near_edge = field.at(&point([0.5, 0.02, 0.02])).value();
    let center = field.at(&point([0.5, 0.5, 0.5])).value();
    assert!((on_edge - 0.5).abs() < 1e-12, "on the edge the size is L/N");
    assert!(near_edge > on_edge && near_edge < 0.6);
    assert!(center > near_edge);
    // Every cube edge is 0.7071 from the centre, so 0.5 + 1.0 * 0.7071.
    assert!((center - (0.5 + 0.5_f64.sqrt())).abs() < 1e-9);
}

#[test]
fn segments_per_edge_scales_the_source() {
    let coarse = FeatureSizing::of(
        &unit_cube(),
        1,
        Some(length(0.01)),
        Some(length(10.0)),
        Some(1.0),
    );
    let fine = FeatureSizing::of(
        &unit_cube(),
        4,
        Some(length(0.01)),
        Some(length(10.0)),
        Some(1.0),
    );
    assert!((coarse.at(&point([0.5, 0.0, 0.0])).value() - 1.0).abs() < 1e-12);
    assert!((fine.at(&point([0.5, 0.0, 0.0])).value() - 0.25).abs() < 1e-12);
}

#[test]
fn respects_the_clamps() {
    let capped = FeatureSizing::of(
        &unit_cube(),
        2,
        Some(length(0.05)),
        Some(length(0.3)),
        Some(1.0),
    );
    assert!((capped.at(&point([0.5, 0.5, 0.5])).value() - 0.3).abs() < 1e-12);
    let floored = FeatureSizing::of(
        &unit_cube(),
        8,
        Some(length(0.4)),
        Some(length(10.0)),
        Some(1.0),
    );
    assert!((floored.at(&point([0.5, 0.0, 0.0])).value() - 0.4).abs() < 1e-12);
}

#[test]
fn unbounded_gradation_is_one_fine_layer() {
    let field = FeatureSizing::of(
        &unit_cube(),
        2,
        Some(length(0.05)),
        Some(length(10.0)),
        None,
    );
    let source = 0.5; // edge length 1.0 over 2 segments
    // Within one target size of an edge: the feature size.
    assert!((field.at(&point([0.5, 0.0, 0.0])).value() - source).abs() < 1e-12);
    assert!((field.at(&point([0.5, source * 0.9, 0.0])).value() - source).abs() < 1e-12);
    // Beyond it: straight to the maximum, no ramp.
    assert!((field.at(&point([0.5, 0.5, 0.5])).value() - 10.0).abs() < 1e-12);
}

#[test]
fn proximity_caps_the_interior_at_the_local_feature_size() {
    let build = || {
        FeatureSizing::of(
            &unit_cube(),
            2,
            Some(length(0.005)),
            Some(length(10.0)),
            None,
        )
    };
    let plain = build();
    let near = build().with_proximity(&unit_cube(), 4).unwrap();
    // Without proximity the crease term leaves the interior at `maximum`.
    assert!((plain.at(&point([0.5, 0.5, 0.5])).value() - 10.0).abs() < 1e-9);
    // With it, every interior point is capped at the through-thickness (1) / 4,
    // wherever it sits along the chord — not at its distance to a face.
    for z in [0.5, 0.2, 0.05] {
        assert!(
            (near.at(&point([0.5, 0.5, z])).value() - 1.0 / 4.0).abs() < 5e-3,
            "z = {z}"
        );
    }
}

#[test]
fn proximity_sees_a_thin_slab() {
    use crate::geometry::cad::brep::test::axis_aligned_box;
    // 0.2 thin in x, thick in y and z.
    let brep = axis_aligned_box([0.2, 4.0, 8.0]);
    let field = FeatureSizing::of(&brep, 2, Some(length(1e-4)), Some(length(10.0)), None)
        .with_proximity(&brep, 4)
        .unwrap();
    // Anywhere through the slab: capped at the 0.2 thickness / 4, not the
    // 4 or 8 spans.
    for x in [0.02, 0.1, 0.18] {
        assert!(
            (field.at(&point([x, 2.0, 4.0])).value() - 0.2 / 4.0).abs() < 5e-3,
            "x = {x}"
        );
    }
}

#[test]
fn proximity_survives_a_face_with_flipped_orientation() {
    use crate::geometry::cad::brep::test::axis_aligned_box;
    let mut brep = axis_aligned_box([0.2, 4.0, 8.0]);
    // The two big faces of the slab (normals along x) claim the opposite
    // orientation, as a real file's inconsistent `same_sense` can leave them.
    for face in [4, 5] {
        brep.faces[face].forward = false;
    }
    let field = FeatureSizing::of(&brep, 2, Some(length(1e-4)), Some(length(10.0)), None)
        .with_proximity(&brep, 4)
        .unwrap();
    assert!(
        (field.at(&point([0.1, 2.0, 4.0])).value() - 0.2 / 4.0).abs() < 5e-3,
        "{}",
        field.at(&point([0.1, 2.0, 4.0])).value()
    );
    assert!(field.proximity_per_face()[4][0] > 0 && field.proximity_per_face()[5][0] > 0);
}

#[test]
fn proximity_sees_a_thin_slab_with_no_minimum() {
    use crate::geometry::cad::brep::test::axis_aligned_box;
    let brep = axis_aligned_box([0.2, 4.0, 8.0]);
    let field = FeatureSizing::of(&brep, 2, None, Some(length(10.0)), None)
        .with_proximity(&brep, 4)
        .unwrap();
    for x in [0.02, 0.1, 0.18] {
        assert!(
            (field.at(&point([x, 2.0, 4.0])).value() - 0.2 / 4.0).abs() < 5e-3,
            "x = {x}: {}",
            field.at(&point([x, 2.0, 4.0])).value()
        );
    }
}

#[test]
fn curvature_resolves_a_bare_cylinder_wall() {
    use crate::geometry::cad::brep::test::capped_cylinder;
    // Radius 1, height 4: mid-height the lateral wall is two radii from either
    // rim, so the crease term has long since ramped to `maximum`.
    let brep = capped_cylinder(1.0, 4.0);
    let plain = FeatureSizing::of(&brep, 2, Some(length(1e-3)), Some(length(10.0)), Some(0.2));
    let curved = FeatureSizing::of(&brep, 2, Some(length(1e-3)), Some(length(10.0)), Some(0.2))
        .with_curvature(&brep, 16)
        .unwrap();
    let on_wall = point([1.0, 0.0, 2.0]);
    let expected = std::f64::consts::TAU / 16.0; // TAU * radius / sections
    assert!(plain.at_cell(&on_wall, 0.05).value() > 4.0 * expected);
    let got = curved.at_cell(&on_wall, 0.05).value();
    assert!(
        got > 0.75 * expected && got < 1.25 * expected,
        "got {got}, expected ~{expected}"
    );
}

#[test]
fn proximity_anchors_a_curved_wall_all_the_way_around() {
    use crate::geometry::cad::brep::test::capped_cylinder;
    // The lateral wall is rotationally symmetric, so proximity must report the
    // same size at every angle around it — the planar path alone leaves an
    // azimuthal gap that a bore then inherits as lopsided refinement.
    let brep = capped_cylinder(1.0, 4.0);
    let plain = FeatureSizing::of(&brep, 2, Some(length(1e-3)), Some(length(10.0)), None);
    let prox = FeatureSizing::of(&brep, 2, Some(length(1e-3)), Some(length(10.0)), None)
        .with_proximity(&brep, 4)
        .unwrap();
    let unfloored = FeatureSizing::of(&brep, 2, None, Some(length(10.0)), None)
        .with_proximity(&brep, 4)
        .unwrap();
    for deg in [0, 45, 90, 135, 180, 225, 270, 315] {
        let t = (deg as f64).to_radians();
        let p = point([t.cos(), t.sin(), 2.0]);
        let size = unfloored.at(&p).value();
        assert!(size < 0.5 && size > 0.05, "no minimum, deg {deg}: {size}");
    }
    for deg in [0, 45, 90, 135, 180, 225, 270, 315] {
        let t = (deg as f64).to_radians();
        let p = point([t.cos(), t.sin(), 2.0]);
        // The planar path never tiles this face; the crease ramp has hit its
        // ceiling two radii from either rim.
        assert!(plain.at(&p).value() > 1.0, "deg {deg}");
        // With the revolved path every angle is anchored to the local
        // through-dimension — no azimuthal gap.
        assert!(
            prox.at(&p).value() < 0.5,
            "deg {deg}: {}",
            prox.at(&p).value()
        );
    }
}

#[test]
fn curvature_leaves_the_far_field_alone() {
    use crate::geometry::cad::brep::test::capped_cylinder;
    let brep = capped_cylinder(1.0, 4.0);
    let curved = FeatureSizing::of(&brep, 2, Some(length(1e-3)), Some(length(10.0)), Some(0.2))
        .with_curvature(&brep, 16)
        .unwrap();
    // Well outside the wall band: back to the crease ramp / `maximum`.
    assert!(curved.at_cell(&point([6.0, 0.0, 2.0]), 0.05).value() > 1.0);
}

#[test]
fn unbounded_max_and_gradation_without_a_thickness_term_is_rejected() {
    use crate::geometry::{cad::brep::test::ball, solid::Solid};
    // maximum: None + gradation: None + no proximity/curvature + no sharp edges:
    // the crease term is INF everywhere, so the octree never refines. The driver
    // must reject the degenerate field, not hand back a 1-node tree.
    let smooth = ball(1.0);
    let empty = FeatureSizing::of(&smooth, 2, Some(length(1e-3)), None, None);
    assert!(smooth.sizing_octree(&empty, Some(4), 0.1).is_err());
    // A cube's sharp edges do drive refinement even with no ramp: the crease
    // term's half-cell slack pulls a fine shell around every edge before the
    // size jumps to `maximum`.
    let brep = unit_cube();
    let creased = FeatureSizing::of(&brep, 2, Some(length(1e-3)), None, None);
    assert!(brep.sizing_octree(&creased, Some(4), 0.1).is_ok());
    // A proximity term anchors the interior too.
    let anchored = FeatureSizing::of(&brep, 2, Some(length(1e-3)), None, None)
        .with_proximity(&brep, 3)
        .unwrap();
    assert!(brep.sizing_octree(&anchored, Some(4), 0.1).is_ok());
}

#[test]
fn arc_polyline_traces_the_true_arc_past_half_a_turn() {
    // 270 deg CCW about +z, r = 1, centre origin: (1,0,0) -> (0,-1,0) the long
    // way, through (-1,0,0). The old shortest-arc rule traced the 90 deg
    // complement through (0.7, -0.7) instead.
    let poly = super::arc_polyline(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        1.0,
        1.0,
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        false,
        12,
    );
    let mid = super::point(&poly[poly.len() / 2]);
    assert!(mid[0] < -0.5, "arc took the short way: {mid:?}");
    let length: f64 = poly
        .windows(2)
        .map(|w| {
            let (a, b) = (super::point(&w[0]), super::point(&w[1]));
            (0..3).map(|k| (b[k] - a[k]).powi(2)).sum::<f64>().sqrt()
        })
        .sum();
    assert!(
        (length - 1.5 * std::f64::consts::PI).abs() < 0.05,
        "length {length}, want ~{}",
        1.5 * std::f64::consts::PI
    );
}

#[test]
fn obeys_the_gradation_bound() {
    let gradation = 0.7;
    let field = FeatureSizing::of(
        &unit_cube(),
        2,
        Some(length(0.05)),
        Some(length(10.0)),
        Some(gradation),
    );
    let samples: [[f64; 3]; 5] = [
        [0.5, 0.0, 0.0],
        [0.5, 0.1, 0.1],
        [0.5, 0.5, 0.5],
        [0.2, 0.3, 0.9],
        [0.9, 0.9, 0.9],
    ];
    for a in samples {
        for b in samples {
            let separation =
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            let difference = (field.at(&point(a)).value() - field.at(&point(b)).value()).abs();
            assert!(
                difference <= gradation * separation + 1e-9,
                "size jumped {difference} over {separation}"
            );
        }
    }
}

/// Two unconnected unit squares in the z = 0 plane, `gap` apart along x and
/// sharing no vertex: square A spans x in [0, 1], square B spans
/// [1 + gap, 2 + gap]. Every edge is incident to only its own face, so
/// `features()` reads all eight as creases (open-shell edges are sharp by
/// the "not exactly two incident faces" rule).
fn two_close_squares(gap: f64) -> Brep {
    let vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0 + gap, 0.0, 0.0],
        [2.0 + gap, 0.0, 0.0],
        [2.0 + gap, 1.0, 0.0],
        [1.0 + gap, 1.0, 0.0],
    ]
    .into_iter()
    .map(Coordinate::const_from)
    .collect();
    let edges = vec![
        edge(0, 1), // 0
        edge(1, 2), // 1  nearest to square B
        edge(2, 3), // 2
        edge(3, 0), // 3
        edge(4, 5), // 4
        edge(5, 6), // 5
        edge(6, 7), // 6
        edge(7, 4), // 7  nearest to square A
    ];
    let up = [0.0, 0.0, 1.0];
    let reference = [1.0, 0.0, 0.0];
    let faces = vec![
        face(up, reference, &[(0, true), (1, true), (2, true), (3, true)]),
        face(up, reference, &[(4, true), (5, true), (6, true), (7, true)]),
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

#[test]
fn feature_separation_resolves_a_narrow_gap_between_unrelated_creases() {
    let brep = two_close_squares(0.1);
    let field = FeatureSizing::of(&brep, 2, Some(length(1e-4)), Some(length(10.0)), None)
        .with_feature_separation(&brep, 4)
        .unwrap();
    // Midpoint of the gap between edge 1 (square A) and edge 7 (square B),
    // exactly 0.1 apart and sharing no vertex: capped at 0.1 / 4, not the
    // crease term's own (edge-length-derived, much coarser) value.
    let mid = point([1.05, 0.5, 0.0]);
    assert!(
        (field.at(&mid).value() - 0.1 / 4.0).abs() < 5.0e-3,
        "{}",
        field.at(&mid).value()
    );
}

#[test]
fn feature_separation_needs_the_builder_to_engage() {
    // Mutation check: without with_feature_separation, nothing pulls that
    // midpoint's target down to the gap size -- the plain crease term sees
    // only its own edges' lengths, both 1 unit, far coarser than the gap.
    let brep = two_close_squares(0.1);
    let plain = FeatureSizing::of(&brep, 2, Some(length(1e-4)), Some(length(10.0)), None);
    let mid = point([1.05, 0.5, 0.0]);
    assert!(plain.at(&mid).value() > 0.1, "{}", plain.at(&mid).value());
}

/// A unit square (its own crease-only shell, as [`two_close_squares`]) sitting
/// `gap` above a much larger floor plate, well inside the floor's own
/// footprint -- so the square's edges pass close to the floor's *surface*
/// while staying far (`>> gap`) from any of the floor's own boundary edges.
/// The nearest unrelated feature here is a face, not another crease.
fn crease_over_a_distant_floor(gap: f64) -> Brep {
    let vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [-10.0, -10.0, -gap],
        [10.0, -10.0, -gap],
        [10.0, 10.0, -gap],
        [-10.0, 10.0, -gap],
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
    ];
    let up = [0.0, 0.0, 1.0];
    let reference = [1.0, 0.0, 0.0];
    let faces = vec![
        face(up, reference, &[(0, true), (1, true), (2, true), (3, true)]),
        face(up, reference, &[(4, true), (5, true), (6, true), (7, true)]),
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

#[test]
fn feature_separation_resolves_a_crease_passing_close_to_an_unrelated_face() {
    let brep = crease_over_a_distant_floor(0.1);
    let field = FeatureSizing::of(&brep, 2, Some(length(1e-4)), Some(length(10.0)), None)
        .with_feature_separation(&brep, 4)
        .unwrap();
    // Midpoint of square A's edge 0: on the crease itself, 0.1 above the
    // floor (directly beneath the whole square) and > 9 units from the
    // floor's own edges: only the crease-to-face measurement can see this.
    let on_crease = point([0.5, 0.0, 0.0]);
    assert!(
        (field.at(&on_crease).value() - 0.1 / 4.0).abs() < 5.0e-3,
        "{}",
        field.at(&on_crease).value()
    );
}

#[test]
fn feature_separation_needs_the_face_query_to_engage() {
    // Mutation check: crease-to-crease alone (the old with_crease_separation
    // behaviour) does not see this gap -- the floor's own edges are >9 units
    // away, so its crease-to-crease minimum stays at the far corner distance,
    // nowhere near 0.1 / 4.
    let brep = crease_over_a_distant_floor(0.1);
    let creases_only = FeatureSizing::of(&brep, 2, Some(length(1e-4)), Some(length(10.0)), None);
    let on_crease = point([0.5, 0.0, 0.0]);
    assert!(
        creases_only.at(&on_crease).value() > 0.2,
        "{}",
        creases_only.at(&on_crease).value()
    );
}

#[test]
fn shares_a_vertex_is_precise() {
    let brep = two_close_squares(0.1);
    // Edge 0 (0,1) and edge 1 (1,2) meet at vertex 1: a real corner.
    assert!(shares_a_vertex(&brep, 0, 1));
    // Edge 1 (1,2) and edge 7 (7,4), the two nearest edges of the separate
    // squares, share no vertex.
    assert!(!shares_a_vertex(&brep, 1, 7));
}
