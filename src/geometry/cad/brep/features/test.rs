use super::super::test::{capped_cylinder, coplanar_squares, unit_cube};

#[test]
fn every_cube_edge_and_corner_is_sharp() {
    let features = unit_cube().features();
    assert_eq!(features.creases, (0..12).collect::<Vec<_>>());
    assert_eq!(features.corners, (0..8).collect::<Vec<_>>());
}

#[test]
fn flat_shared_edge_is_not_a_crease() {
    let features = coplanar_squares().features();
    assert_eq!(features.creases, vec![0, 2, 3, 4, 5, 6]);
    assert_eq!(features.corners, vec![0, 2, 3, 5]);
}

#[test]
fn a_closed_loop_crease_has_no_corner_at_its_seam() {
    // Each rim of a capped cylinder is a single circular edge with coincident
    // start/end vertices (the seam where the parametrisation cuts, not a real
    // discontinuity) and a genuinely sharp cap/wall dihedral either side of it.
    let brep = capped_cylinder(2.0, 5.0);
    let features = brep.features();
    assert_eq!(features.creases.len(), 2, "both rims are sharp creases");
    assert!(
        features.corners.is_empty(),
        "smooth seam vertices reported as corners: {:?}",
        features.corners
    );
}

#[test]
fn dihedral_crosses_the_cutoff() {
    // Fold the right square about the shared edge (which runs along y). Under the
    // 30-degree deviation cutoff the fold stays smooth; past it, it is a crease.
    let fold = |degrees: f64| {
        let mut brep = coplanar_squares();
        let angle = degrees.to_radians();
        let super::super::surface::Surface::Plane(plane) = &mut brep.faces[1].surface else {
            unreachable!()
        };
        plane.normal = crate::geometry::Direction::const_from([angle.sin(), 0.0, angle.cos()]);
        brep.features().creases.contains(&1)
    };
    assert!(!fold(15.0));
    assert!(fold(45.0));
}
