use crate::geometry::cad::brep::test::{capped_cylinder, unit_cube};

#[test]
fn a_cube_has_twelve_straight_crease_curves() {
    let curves = unit_cube().crease_curves();
    assert_eq!(curves.len(), 12);
    for (curve, faces) in &curves {
        assert_eq!(
            curve.len(),
            2,
            "a straight edge chords to its two endpoints exactly"
        );
        assert_eq!(faces.len(), 2, "a cube edge borders exactly two faces");
    }
}

#[test]
fn a_closed_rim_crease_curve_closes_up() {
    // Each rim of a capped cylinder is one circular edge with coincident
    // start/end vertices; its chord polyline should still trace the full turn.
    let curves = capped_cylinder(2.0, 5.0).crease_curves();
    assert_eq!(curves.len(), 2);
    for (curve, faces) in &curves {
        assert!(
            curve.len() > 8,
            "a circle needs more than a couple of chords"
        );
        assert_eq!(faces.len(), 2, "a rim borders the cap and the wall");
        let (first, last) = (&curve[0], curve.last().unwrap());
        let gap = (0..3)
            .map(|k| (first[k].value() - last[k].value()).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(gap < 1.0e-9, "polyline does not close: gap {gap}");
    }
}
