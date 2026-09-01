use crate::{
    geometry::{
        Coordinate,
        cad::brep::{
            curve::Ellipse,
            test::{
                ball, bulged_plate, capped_cylinder, cone, cube_with_bspline_top,
                cylinder_with_elliptical_rim, direction,
                cone_split_at_apex, cylinder_with_splined_rim, hemisphere_solid,
                partial_cone_to_apex,
                cylinder_with_slanted_edge, partial_cylinder, partial_sphere, partial_torus,
                revolved_cylinder,
                square_with_rounded_hole, square_with_splined_hole, torus, unit_cube,
            },
        },
        solid::SolidOracle,
    },
    math::TensorRank1,
};
use std::array::from_fn;

fn close(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12)
}

fn components<I, U>(tensor: &TensorRank1<3, I, U>) -> [f64; 3] {
    from_fn(|k| tensor[k].value())
}

#[test]
fn ray_parity_signs_a_capped_cylinder() {
    let oracle = capped_cylinder(2.0, 5.0).oracle().unwrap();
    let sign = |p: [f64; 3]| oracle.signed_distance(&Coordinate::from(p)).is_sign_positive();
    assert!(sign([0.0, 0.0, 2.5]), "axis mid-height is inside");
    assert!(sign([1.9, 0.0, 2.5]), "just inside the wall is inside");
    assert!(!sign([2.1, 0.0, 2.5]), "just outside the wall is outside");
    assert!(!sign([0.0, 0.0, 6.0]), "above the top cap is outside");
    assert!(!sign([0.0, 0.0, -1.0]), "below the base is outside");
    // A point in the cylinder's "shadow" but well outside it — nearest-face
    // normal could pick the cap and mis-sign; ray parity does not.
    assert!(!sign([0.0, 0.0, 20.0]));
}

#[test]
fn projects_an_exterior_point_onto_the_nearest_face() {
    let oracle = unit_cube().oracle().unwrap();
    let (point, normal) = oracle.project(&Coordinate::from([0.4, 0.6, 1.3])).unwrap();
    assert!(close(&components(&point), &[0.4, 0.6, 1.0]));
    assert!(close(&components(&normal), &[0.0, 0.0, 1.0]));
}

#[test]
fn projects_an_interior_point_onto_the_nearest_face() {
    let oracle = unit_cube().oracle().unwrap();
    let (point, normal) = oracle.project(&Coordinate::from([0.4, 0.6, 0.85])).unwrap();
    assert!(close(&components(&point), &[0.4, 0.6, 1.0]));
    assert!(close(&components(&normal), &[0.0, 0.0, 1.0]));
}

#[test]
fn a_bspline_face_projects_and_signs_like_the_plane_it_lies_on() {
    let oracle = cube_with_bspline_top().oracle().unwrap();

    // The free-form top is flat at z = 1, so projection onto it is exact.
    let (point, normal) = oracle.project(&Coordinate::from([0.4, 0.7, 2.0])).unwrap();
    let point = components(&point);
    assert!((point[2] - 1.0).abs() < 1.0e-6, "{point:?}");
    assert!((point[0] - 0.4).abs() < 1.0e-3 && (point[1] - 0.7).abs() < 1.0e-3, "{point:?}");
    assert!(components(&normal)[2].abs() > 0.999, "{:?}", components(&normal));

    // Ray parity still classifies the solid with a free-form face in the shell.
    let inside = |p: [f64; 3]| oracle.signed_distance(&Coordinate::from(p)).is_sign_positive();
    assert!(inside([0.5, 0.5, 0.5]), "cube centre is inside");
    assert!(!inside([0.5, 0.5, 1.5]), "above the free-form top is outside");
    assert!(!inside([0.5, 0.5, -0.5]), "below the base is outside");
    let near_top = oracle.signed_distance(&Coordinate::from([0.5, 0.5, 0.9]));
    assert!((near_top - 0.1).abs() < 1.0e-3, "{near_top}");
}

#[test]
fn a_surface_of_revolution_matches_the_cylinder_it_describes() {
    let oracle = revolved_cylinder(2.0, 5.0).oracle().unwrap();

    // Projection onto the revolved wall lands on the cylinder of radius 2.
    let (point, normal) = oracle.project(&Coordinate::from([5.0, 0.0, 2.5])).unwrap();
    let point = components(&point);
    assert!((point[0].hypot(point[1]) - 2.0).abs() < 5.0e-3, "{point:?}");
    assert!((point[2] - 2.5).abs() < 5.0e-3, "{point:?}");
    assert!(components(&normal)[0] > 0.99, "{:?}", components(&normal));

    let inside = |p: [f64; 3]| oracle.signed_distance(&Coordinate::from(p)).is_sign_positive();
    assert!(inside([0.0, 0.0, 2.5]), "axis is inside");
    assert!(inside([1.9, 0.0, 2.5]), "just inside the wall");
    assert!(!inside([2.1, 0.0, 2.5]), "just outside the wall");
    assert!(!inside([0.0, 0.0, 6.0]), "above the top cap");
    assert!(!inside([0.0, 0.0, 20.0]), "far down the shadow is still outside");
}

#[test]
fn a_non_axial_straight_edge_trims_the_cylinder_instead_of_refusing_it() {
    // The slanted top edge used to make `oracle()` error outright.
    let oracle = cylinder_with_slanted_edge(2.0, 4.0, 1.2).oracle().unwrap();

    // A point outside the wall, mid-wedge, projects onto the cylinder r = 2.
    let mid = [2.5 * 0.6_f64.cos(), 2.5 * 0.6_f64.sin(), 2.0];
    let (point, _) = oracle.project(&Coordinate::from(mid)).unwrap();
    let p = components(&point);
    assert!((p[0].hypot(p[1]) - 2.0).abs() < 1.0e-6, "{p:?}");
    assert!((p[1].atan2(p[0]) - 0.6).abs() < 1.0e-3, "mid-wedge angle {}", p[1].atan2(p[0]));

    // A point in the angular gap clamps back to the θ = 1.2 trim edge, not
    // onto the untrimmed cylinder at its own angle.
    let gap = [2.0 * 2.0_f64.cos(), 2.0 * 2.0_f64.sin(), 2.0];
    let (point, _) = oracle.project(&Coordinate::from(gap)).unwrap();
    let p = components(&point);
    assert!(p[1].atan2(p[0]) < 1.4, "gap point clamps to the trim edge, got {}", p[1].atan2(p[0]));
}

#[test]
fn a_point_on_a_face_projects_to_itself() {
    let oracle = unit_cube().oracle().unwrap();
    let (point, normal) = oracle
        .project(&Coordinate::from([0.25, 0.75, 0.0]))
        .unwrap();
    assert!(close(&components(&point), &[0.25, 0.75, 0.0]));
    assert!(close(&components(&normal), &[0.0, 0.0, -1.0]));
}

#[test]
fn clamps_a_point_past_an_edge_to_the_trimming_loop() {
    let oracle = unit_cube().oracle().unwrap();
    let (point, _) = oracle.project(&Coordinate::from([1.3, 0.5, 1.1])).unwrap();
    assert!(close(&components(&point), &[1.0, 0.5, 1.0]));
}

#[test]
fn clamps_a_point_past_a_corner_to_the_vertex() {
    let oracle = unit_cube().oracle().unwrap();
    let (point, _) = oracle.project(&Coordinate::from([1.4, 1.2, 1.3])).unwrap();
    assert!(close(&components(&point), &[1.0, 1.0, 1.0]));
}

#[test]
fn rejects_a_brep_with_no_faces() {
    let mut brep = unit_cube();
    brep.faces.clear();
    assert!(brep.oracle().is_err());
}

#[test]
fn signs_a_capped_cylinder_from_the_nearest_face() {
    let oracle = capped_cylinder(2.0, 5.0).oracle().unwrap();
    // Positive inside, magnitude the distance to the nearer wall/cap.
    assert!((oracle.signed_distance(&Coordinate::from([0.0, 0.0, 2.5])) - 2.0).abs() < 1e-9);
    assert!((oracle.signed_distance(&Coordinate::from([0.0, 0.0, 0.5])) - 0.5).abs() < 1e-9);
    // Negative outside.
    assert!((oracle.signed_distance(&Coordinate::from([5.0, 0.0, 2.5])) + 3.0).abs() < 1e-9);
    assert!((oracle.signed_distance(&Coordinate::from([0.0, 0.0, -1.0])) + 1.0).abs() < 1e-9);
}

#[test]
fn projects_onto_a_cylindrical_wall() {
    let oracle = capped_cylinder(2.0, 5.0).oracle().unwrap();
    let (point, normal) = oracle.project(&Coordinate::from([3.0, 0.0, 2.0])).unwrap();
    assert!(close(&components(&point), &[2.0, 0.0, 2.0]));
    assert!(close(&components(&normal), &[1.0, 0.0, 0.0]));
}

#[test]
fn accepts_a_full_cone_face() {
    assert!(cone(3.0, 1.0, 4.0).oracle().is_ok());
}

#[test]
fn ellipse_sinusoid_matches_the_plane_cylinder_intersection() {
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let ellipse = Ellipse {
        center: Coordinate::from([0.0, 0.0, 5.0]),
        axis: direction([0.0, h, h]),
        reference_direction: direction([1.0, 0.0, 0.0]),
        major_radius: 2.0 * std::f64::consts::SQRT_2,
        minor_radius: 2.0,
    };
    let sinusoid = super::ellipse_sinusoid(&ellipse, [0.0; 3], [0.0, 0.0, 1.0], 2.0).unwrap();
    for u in [0.0_f64, 0.7, 1.5, -1.0] {
        assert!((sinusoid.v(u) - (5.0 - 2.0 * u.sin())).abs() < 1e-9);
    }
}

#[test]
fn accepts_a_cylindrical_face_with_a_tilted_elliptical_rim() {
    assert!(
        cylinder_with_elliptical_rim(2.0, std::f64::consts::FRAC_PI_3)
            .oracle()
            .is_ok()
    );
}

#[test]
fn a_tilted_elliptical_rim_over_half_a_turn_still_closes_the_trim() {
    // angle > pi: `wrap` alone would collapse the ellipse edge onto the wrong
    // branch and the ring would not close. The axial component of the cut
    // plane's normal picks the branch.
    let angle = 4.0_f64;
    let oracle = cylinder_with_elliptical_rim(2.0, angle).oracle().unwrap();
    // A point sitting exactly on the trimmed wall, mid-sweep: it must not move.
    let u = angle / 2.0;
    let z = 5.0 - 2.0 * u.sin();
    let query = Coordinate::from([2.0 * u.cos(), 2.0 * u.sin(), z]);
    let (point, _) = oracle.project(&query).unwrap();
    let moved = components(&point)
        .iter()
        .zip([2.0 * u.cos(), 2.0 * u.sin(), z])
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(moved < 1.0e-6, "on-surface point moved {moved}");
}

#[test]
fn a_tilted_elliptical_rim_snaps_onto_the_true_cut_not_a_flat_one() {
    let query_angle = std::f64::consts::FRAC_PI_6; // well inside [0, pi/3]
    let oracle = cylinder_with_elliptical_rim(2.0, std::f64::consts::FRAC_PI_3)
        .oracle()
        .unwrap();
    // On the infinite cylinder, well past the true (tilted) cut everywhere
    // over [0, pi/3]: a flat/full wall would leave this point exactly where
    // it is (already on the surface, no trim at all).
    let query = Coordinate::from([2.0 * query_angle.cos(), 2.0 * query_angle.sin(), 6.0]);
    let (point, _) = oracle.project(&query).unwrap();
    let [x, y, z] = components(&point);
    // The nearest point on a sloped curve doesn't preserve the query's own
    // angle, so check self-consistently: it lands exactly on the true
    // sinusoid at whatever angle it actually snapped to.
    assert!((z - (5.0 - 2.0 * y.atan2(x).sin())).abs() < 1e-6);
    // And it moved — the query was already sitting on the untrimmed surface.
    assert!((z - 6.0).abs() > 0.1);
}

#[test]
fn nearest_on_sinusoid_finds_the_global_minimum_not_a_saddle() {
    // v(u) = 3 cos(u), query (0, 0): u = 0 is a critical point of the squared
    // distance (d = 3) but not the minimum, which is near u = +-1.41 (d ~ 1.49).
    // The old single-bracket bisection stopped at u = 0.
    let sinusoid = super::Sinusoid { k: 0.0, a: 3.0, phi: 0.0 };
    let [u, v] = super::nearest_on_sinusoid([0.0, 0.0], -3.0, 3.0, &sinusoid);
    let distance = (u * u + v * v).sqrt();
    assert!(distance < 1.6, "landed on a non-minimal critical point: u = {u}, d = {distance}");
}

#[test]
fn accepts_a_planar_face_with_a_rounded_rectangle_hole() {
    assert!(square_with_rounded_hole().oracle().is_ok());
}

#[test]
fn a_planar_patch_box_covers_a_bulging_arc_edge() {
    let brep = bulged_plate();
    let face = brep.planar_face(&brep.faces[0]).unwrap();
    let (low, high) = super::patch::FacePatch::Planar(face).bounds();
    // The arc bulges to y = 6; the loop vertices only reach y = 4.
    assert!(high[1] >= 6.0 - 1e-9, "arc bulge to y=6 not in the box: {low:?}..{high:?}");
}

#[test]
fn accepts_a_partial_spherical_face() {
    assert!(partial_sphere(2.0).oracle().is_ok());
}

#[test]
fn a_hemisphere_has_no_phantom_surface_below_its_rim() {
    // The equator runs CCW about +z on an outward-facing face, so the trimmed
    // patch is the northern cap; the southern half must not be there.
    let oracle = partial_sphere(2.0).oracle().unwrap();
    let query = Coordinate::from([1.0, 0.0, -3.0]);
    let (point, _) = oracle.project(&query).unwrap();
    let [x, y, z] = components(&point);
    // A whole sphere would answer with the radial projection, well below z = 0.
    assert!(z.abs() < 1.0e-6, "landed at z = {z}, not on the rim");
    assert!((x.hypot(y) - 2.0).abs() < 1.0e-6);
    assert!(y.atan2(x).abs() < 1.0e-6, "rim point drifted off the query meridian");
    // And a point over the kept cap still projects radially onto it.
    let north = Coordinate::from([0.0, 0.0, 3.0]);
    let (point, _) = oracle.project(&north).unwrap();
    assert!(close(&components(&point), &[0.0, 0.0, 2.0]));
}

/// The same patch with every half-edge walked backwards. The reader orients a
/// circle's axis for its edge's own direction, so a reversed half-edge has to
/// turn about the negated axis; taking the positive turn regardless sends the
/// chord the long way round and the ring gains a whole spurious turn.
#[test]
fn a_toroidal_trim_survives_a_loop_walked_backwards() {
    let angle = std::f64::consts::FRAC_PI_2;
    let mut brep = partial_torus(4.0, 1.5, angle);
    let half_edges = &mut brep.faces[0].bounds[0].half_edges;
    half_edges.reverse();
    half_edges.iter_mut().for_each(|half_edge| half_edge.forward = false);
    let oracle = brep.oracle().expect("reversed loop did not close");
    let (major, minor) = (4.0, 1.5);
    let past = angle + 0.5;
    let tube = std::f64::consts::FRAC_PI_4;
    let on_surface = [
        (major + minor * tube.cos()) * past.cos(),
        (major + minor * tube.cos()) * past.sin(),
        minor * tube.sin(),
    ];
    let (point, _) = oracle.project(&Coordinate::from(on_surface)).unwrap();
    let [x, y, _] = components(&point);
    assert!((y.atan2(x) - angle).abs() < 1.0e-6, "reversed trim lost the u rim");
}

#[test]
fn accepts_a_conical_face_closing_to_its_apex() {
    assert!(
        partial_cone_to_apex(2.0, 5.0, std::f64::consts::FRAC_PI_2)
            .oracle()
            .is_ok()
    );
}

#[test]
fn a_cone_wedge_keeps_the_whole_patch_beside_its_apex() {
    // The apex is the whole chart line `v = 0`, not one point on it: a ring
    // that holds its angle across it cuts the corner and loses half the wedge.
    let angle = std::f64::consts::FRAC_PI_2;
    let (radius, height) = (2.0, 5.0);
    let oracle = partial_cone_to_apex(radius, height, angle).oracle().unwrap();
    // Mid-wedge and near the apex, on the surface: it must stay put.
    for fraction in [0.15, 0.5, 0.85] {
        let a = angle * fraction;
        let level = 0.2;
        let r = radius * level;
        let on_surface = [r * a.cos(), r * a.sin(), height * (1.0 - level)];
        let (point, _) = oracle.project(&Coordinate::from(on_surface)).unwrap();
        let moved = components(&point)
            .iter()
            .zip(on_surface)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(moved < 1.0e-6, "point at angle {a} near the apex moved {moved}");
    }
    // Outside the wedge there is no surface to sit on.
    let a = angle + 0.6;
    let query = [radius * a.cos(), radius * a.sin(), 0.0];
    let (point, _) = oracle.project(&Coordinate::from(query)).unwrap();
    let [x, y, _] = components(&point);
    assert!(y.atan2(x) <= angle + 1.0e-6, "kept a phantom wedge past the ruling");
}

#[test]
fn accepts_a_partial_toroidal_face() {
    assert!(partial_torus(4.0, 1.5, std::f64::consts::FRAC_PI_2).oracle().is_ok());
}

#[test]
fn a_partial_toroidal_face_has_no_phantom_tube_in_the_gap() {
    let angle = std::f64::consts::FRAC_PI_2;
    let oracle = partial_torus(4.0, 1.5, angle).oracle().unwrap();
    // Sitting exactly on the untrimmed tube, a quarter turn past the patch: a
    // phantom full torus would leave this point where it is.
    let past = angle + 0.5;
    let (major, minor) = (4.0, 1.5);
    let tube = std::f64::consts::FRAC_PI_4;
    let on_surface = [
        (major + minor * tube.cos()) * past.cos(),
        (major + minor * tube.cos()) * past.sin(),
        minor * tube.sin(),
    ];
    let (point, _) = oracle.project(&Coordinate::from(on_surface)).unwrap();
    let [x, y, _] = components(&point);
    assert!(
        (y.atan2(x) - angle).abs() < 1.0e-6,
        "did not snap back to the u = {angle} rim"
    );
    // Below the outer equator is off the patch in the tube direction too.
    let under = [major + minor, 0.0, -0.5];
    let (point, _) = oracle.project(&Coordinate::from(under)).unwrap();
    assert!(components(&point)[2] > -1.0e-6, "kept a phantom lower tube");
}

#[test]
fn accepts_a_whole_sphere_and_torus() {
    assert!(ball(2.0).oracle().is_ok());
    assert!(torus(4.0, 1.5).oracle().is_ok());
}

#[test]
fn ray_distance_reaches_a_slender_torus_tube() {
    // major/minor = 250. The old fixed 96-step march over the ~10-long chord
    // stepped clean over the ~0.035-wide tube on both sides and ray_distance
    // returned None. The ray is offset in z so its crossings sit mid-interval,
    // not on the bounding-sphere tangent point.
    let oracle = torus(5.0, 0.02).oracle().unwrap();
    let hit = oracle.ray_distance(&Coordinate::from([-10.0, 0.0, 0.01]), [1.0, 0.0, 0.0]);
    assert!(hit.is_some(), "ray stepped over the slender tube");
    // Enters the near tube where sqrt((x+5)^2 + 0.01^2) = 0.02, i.e. at
    // x = -5 - sqrt(3e-4); t = x + 10.
    let want = 5.0 - (3.0e-4_f64).sqrt();
    assert!((hit.unwrap() - want).abs() < 1.0e-3, "got {hit:?}, want ~{want:.4}");
}

#[test]
fn accepts_a_partial_cylindrical_face() {
    assert!(partial_cylinder(2.0, 5.0, std::f64::consts::FRAC_PI_2).oracle().is_ok());
}

#[test]
fn a_partial_cylindrical_face_has_no_phantom_wall_in_the_gap() {
    let angle = std::f64::consts::FRAC_PI_2;
    let oracle = partial_cylinder(2.0, 5.0, angle).oracle().unwrap();
    // Just past the trimmed edge, still well inside the untrimmed gap: a
    // phantom full wall would report this point as already on the surface.
    let query_angle = angle + 0.5;
    let query = Coordinate::from([2.0 * query_angle.cos(), 2.0 * query_angle.sin(), 2.5]);
    let (point, _) = oracle.project(&query).unwrap();
    let [x, y, _] = components(&point);
    assert!((y.atan2(x) - angle).abs() < 1e-6);
}

#[test]
fn axial_span_rejects_a_face_with_no_vertices() {
    assert!(super::axial_span(&[], [0.0; 3], [0.0, 0.0, 1.0]).is_err());
}

#[test]
fn axial_span_rejects_a_degenerate_zero_height_face() {
    let points: Vec<[f64; 3]> = vec![[1.0, 0.0, 3.0], [-1.0, 0.0, 3.0], [0.0, 1.0, 3.0]];
    assert!(super::axial_span(&points, [0.0; 3], [0.0, 0.0, 1.0]).is_err());
}

#[test]
fn accepts_a_cylindrical_face_with_a_free_form_rim() {
    assert!(
        cylinder_with_splined_rim(2.0, std::f64::consts::FRAC_PI_2)
            .oracle()
            .is_ok()
    );
}

#[test]
fn a_b_spline_rim_trims_to_the_true_curve_not_a_flat_one() {
    let angle = std::f64::consts::FRAC_PI_2;
    let oracle = cylinder_with_splined_rim(2.0, angle).oracle().unwrap();
    // Mid-sweep, well above the rim: it must fall onto `z = 5 - sin(u)/2`,
    // which at u = pi/4 sits 0.35 below the flat cut a chord-free trim gives.
    let u = angle / 2.0;
    let query = Coordinate::from([2.0 * u.cos(), 2.0 * u.sin(), 8.0]);
    let (point, _) = oracle.project(&query).unwrap();
    let [x, y, z] = components(&point);
    assert!((z - (5.0 - 0.5 * y.atan2(x).sin())).abs() < 0.01, "landed at z = {z}");
    // A point already on the trimmed wall must not move.
    let inside = Coordinate::from([2.0 * u.cos(), 2.0 * u.sin(), 2.0]);
    let (point, _) = oracle.project(&inside).unwrap();
    assert!(close(&components(&point), &[2.0 * u.cos(), 2.0 * u.sin(), 2.0]));
}

#[test]
fn accepts_a_planar_face_with_a_b_spline_hole() {
    let brep = square_with_splined_hole();
    let face = brep.planar_face(&brep.faces[0]).unwrap();
    assert!(brep.oracle().is_ok());
    // The rational quarter-circles bound a radius-2 hole about (5, 5).
    assert!(!face.contains([5.0, 5.0]));
    assert!(!face.contains([6.4, 5.0]));
    assert!(face.contains([7.5, 5.0]));
    assert!(face.contains([5.0, 8.0]));
    let boundary = face.nearest_boundary([5.0, 5.0 + 1.0]);
    let radius = ((boundary[0] - 5.0).powi(2) + (boundary[1] - 5.0).powi(2)).sqrt();
    assert!((radius - 2.0).abs() < 0.01, "nearest hole boundary at radius {radius}");
}



/// The two halves of an apex cone must tile it, not overlap. Crossing the
/// apex, `wrap` can only offer the shorter turn, so both halves used to swing
/// onto the same side and every ray through the cone was counted twice.
#[test]
fn the_halves_of_an_apex_cone_do_not_overlap() {
    let brep = cone_split_at_apex(2.0, 5.0, 2.0 * std::f64::consts::PI / 3.0);
    let oracle = brep.oracle().unwrap();
    let direction = [0.862_667, 0.411_988, 0.291_536];
    let mut seen = 0;
    for step in 0..40 {
        let query = Coordinate::from([-6.0, -3.0 + 0.15 * step as f64, 1.0]);
        let hits = oracle.ray_report(&query, direction);
        for pair in hits.windows(2) {
            assert!(
                (pair[1].2 - pair[0].2).abs() > 1.0e-9,
                "two faces claim the same crossing at t = {}",
                pair[0].2
            );
        }
        seen += hits.len();
    }
    assert!(seen > 0, "the sweep never crossed the cone");
    // And between them the halves have to cover the cone: every point on the
    // lateral surface belongs to one of them, so projection leaves it alone.
    for spoke in 0..24 {
        let angle = std::f64::consts::TAU * (spoke as f64 + 0.5) / 24.0;
        for level in 1..5 {
            let z = 5.0 * level as f64 / 5.0 - 0.5;
            let r = 2.0 * (1.0 - z / 5.0);
            let on_surface = [r * angle.cos(), r * angle.sin(), z];
            let (point, _) = oracle.project(&Coordinate::from(on_surface)).unwrap();
            let moved = components(&point)
                .iter()
                .zip(on_surface)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(moved < 1.0e-9, "cone surface point at angle {angle} z {z} moved {moved}");
        }
    }
}

/// A plane and a sphere sharing a circular edge: the plane carries it exactly,
/// the sphere can only chord it in its own chart, so the two trimmed regions
/// part company by the chord's sagitta. A ray threading that gap hits neither
/// face, or both, and must not be left to settle the parity on its own — the
/// signed distance would flip sign across a hair's breadth of empty space.
#[test]
fn a_chorded_shared_edge_does_not_flip_the_sign() {
    assert_eq!(sign_flips(&hemisphere_solid(2.0)), 0);
}

/// Random neighbouring pairs whose signed distances disagree in sign with no
/// surface between them — the signature of a boundary a ray can slip through.
fn sign_flips(brep: &crate::geometry::cad::brep::Brep) -> usize {
    let oracle = brep.oracle().unwrap();
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut rand = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let step = 4.0e-4;
    let mut flips = 0;
    for _ in 0..400000 {
        let p: [f64; 3] = from_fn(|_| -3.0 + 6.0 * rand());
        let mut q = p;
        q[(rand() * 3.0) as usize % 3] += step;
        let (a, b) = (
            oracle.signed_distance(&Coordinate::from(p)),
            oracle.signed_distance(&Coordinate::from(q)),
        );
        // A sign change over one step is only honest within a step of a face.
        if (a > 0.0) != (b > 0.0) && a.abs().min(b.abs()) > step {
            flips += 1;
        }
    }
    flips
}


