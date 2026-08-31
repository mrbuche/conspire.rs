use crate::{
    geometry::{
        Coordinate,
        cad::brep::{
            curve::Ellipse,
            test::{
                ball, bulged_plate, capped_cylinder, cone, cylinder_with_elliptical_rim, direction,
                partial_cylinder, partial_sphere, square_with_rounded_hole, torus, unit_cube,
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
fn rejects_a_partial_spherical_face() {
    assert!(partial_sphere(2.0).oracle().is_err());
}

#[test]
fn accepts_a_whole_sphere_and_torus() {
    assert!(ball(2.0).oracle().is_ok());
    assert!(torus(4.0, 1.5).oracle().is_ok());
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
