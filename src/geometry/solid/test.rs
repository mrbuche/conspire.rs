use super::{
    SolidOracle, Uniform, classify_by_flood_fill, classify_by_signed_distance, refine_octree,
    survives_trim,
};
use crate::{
    geometry::{
        Coordinate, Coordinates, Direction,
        mesh::{Class, Connectivity, Mesh},
    },
    math::{Quantity, Scalar},
};

/// The box `1.5 <= x,y,z <= 3.5` (positive inside). With `lie`, the sign is
/// flipped strictly inside `2 < x,y,z < 3` — a stand-in for a
/// nearest-face-normal oracle that is right near the surface but wrong deep in
/// the interior.
struct Box {
    lie: bool,
}

impl SolidOracle for Box {
    fn project(&self, _query: &Coordinate<3>) -> Option<(Coordinate<3>, Direction<3>)> {
        None
    }
    fn signed_distance(&self, query: &Coordinate<3>) -> Scalar {
        let distance = (0..3)
            .map(|k| {
                let v = query[k].value();
                (v - 1.5).min(3.5 - v)
            })
            .fold(Scalar::INFINITY, Scalar::min);
        let deep = (0..3).all(|k| {
            let v = query[k].value();
            v > 2.0 && v < 3.0
        });
        if self.lie && deep {
            -distance
        } else {
            distance
        }
    }
}

/// `n`-cubed unit hexes filling `[0, n]^3`.
fn cube_of_hexes(n: usize) -> Mesh<3> {
    let side = n + 1;
    let mut coordinates = Vec::new();
    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                coordinates.push(Coordinate::from([i as Scalar, j as Scalar, k as Scalar]));
            }
        }
    }
    let node = |i: usize, j: usize, k: usize| (i * side + j) * side + k;
    let mut hexes = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                hexes.push([
                    node(i, j, k),
                    node(i + 1, j, k),
                    node(i + 1, j + 1, k),
                    node(i, j + 1, k),
                    node(i, j, k + 1),
                    node(i + 1, j, k + 1),
                    node(i + 1, j + 1, k + 1),
                    node(i, j + 1, k + 1),
                ]);
            }
        }
    }
    (
        vec![Connectivity::Hexahedral(hexes.into())],
        Coordinates::from(coordinates),
    )
        .into()
}

/// Index of hex `(i, j, k)` in an `n`-cubed grid.
fn hex(n: usize, i: usize, j: usize, k: usize) -> usize {
    (i * n + j) * n + k
}

#[test]
fn flood_fill_carves_the_interior_box() {
    let mesh = cube_of_hexes(5);
    let classes = classify_by_flood_fill(&Box { lie: false }, &mesh).unwrap();
    // The solid occupies hexes 2..=2 on each axis (x in [2,3] etc.), wrapped
    // in a straddle shell, wrapped in air.
    assert_eq!(classes[hex(5, 2, 2, 2)], Class::Inside);
    assert_eq!(classes[hex(5, 1, 2, 2)], Class::Cut);
    assert_eq!(classes[hex(5, 0, 2, 2)], Class::Outside);
    assert_eq!(classes[hex(5, 0, 0, 0)], Class::Outside);
    assert_eq!(classes.iter().filter(|&&c| c == Class::Inside).count(), 1);
}

#[test]
fn origin_snap_keeps_the_geometry_inside_the_root() {
    // Off-origin bbox, zero padding, coarse levels: the origin snap wants to
    // shift the box by up to half a finest cell, and there is no padding margin
    // to absorb it. The shift must be clamped so the geometry stays enclosed.
    let low = Coordinate::from([0.3, 0.3, 0.3]);
    let high = Coordinate::from([1.3, 1.3, 1.3]);
    // A size that forces a level or two of refinement (an unrefined 1-node
    // tree is rejected as a degenerate field), though only the root placement
    // set before refinement matters here.
    let tree = refine_octree(
        (low.clone(), high.clone()),
        &Uniform(Quantity::new(0.1)),
        Some(3),
        0.0,
    )
    .unwrap();
    let rescale = tree.rescale();
    let extent = rescale.cell.value() * rescale.half;
    for k in 0..3 {
        let root_low = rescale.center[k].value() - extent;
        let root_high = rescale.center[k].value() + extent;
        assert!(
            root_low <= low[k].value() + 1e-12 && root_high >= high[k].value() - 1e-12,
            "axis {k}: root [{root_low}, {root_high}] does not enclose geometry [{}, {}]",
            low[k].value(),
            high[k].value(),
        );
    }
}

#[test]
fn flood_fill_ignores_a_lie_in_the_interior() {
    let mesh = cube_of_hexes(5);
    let centre = hex(5, 2, 2, 2);
    // The plain sign test trusts the centroid and mislabels the middle hex.
    let naive = classify_by_signed_distance(&Box { lie: true }, &mesh).unwrap();
    assert_eq!(naive[centre], Class::Outside);
    // The flood fill only trusts the straddle band and the boundary seed.
    let flooded = classify_by_flood_fill(&Box { lie: true }, &mesh).unwrap();
    assert_eq!(flooded[centre], Class::Inside);
}

#[test]
fn trim_keeps_a_flood_fill_rescued_cut_cell() {
    // A `Cut` cell with every corner in the air (maximum < 0) is a thin-wall
    // rescue: Tong's ratio test drops it, `mesh`'s trim must not.
    assert!(survives_trim(true, -0.3, -0.1));
    // Same corners, not flagged `Cut` (an ordinary far-outside cell): dropped.
    assert!(!survives_trim(false, -0.3, -0.1));
    // An ordinary straddling `Cut` cell still obeys the ratio rule.
    assert!(survives_trim(true, -0.05, 0.9));
    assert!(!survives_trim(true, -1.0, 0.5));
}

/// A plate `0.5 <= x,z <= 4.5`, `1.3 <= y <= 1.7` (positive inside): thinner
/// than one unit cell, and placed so no cell corner ever lands in it.
struct Plate;

impl SolidOracle for Plate {
    fn project(&self, _query: &Coordinate<3>) -> Option<(Coordinate<3>, Direction<3>)> {
        None
    }
    fn signed_distance(&self, query: &Coordinate<3>) -> Scalar {
        let (low, high) = ([0.5, 1.3, 0.5], [4.5, 1.7, 4.5]);
        (0..3)
            .map(|k| {
                let v = query[k].value();
                (v - low[k]).min(high[k] - v)
            })
            .fold(Scalar::INFINITY, Scalar::min)
    }
}

#[test]
fn flood_fill_keeps_a_wall_thinner_than_its_cell() {
    let mesh = cube_of_hexes(5);
    let classes = classify_by_flood_fill(&Plate, &mesh).unwrap();
    // Every corner of the y = 1..2 cell layer is in the air (the plate spans
    // only 1.3..1.7), so the eight-corner sign test alone reads the whole
    // layer as unanimously outside and deletes the plate.
    for i in 1..4 {
        for k in 1..4 {
            assert_eq!(
                classes[hex(5, i, 1, k)],
                Class::Cut,
                "plate cell ({i}, 1, {k}) was dropped"
            );
        }
    }
    // The rescue is not a blanket one: cells with no solid in them at all stay
    // `Outside`, above and below the plate and off its footprint.
    for (i, j, k) in [(2, 0, 2), (2, 2, 2), (2, 3, 2), (0, 1, 0), (4, 1, 4)] {
        assert_eq!(
            classes[hex(5, i, j, k)],
            Class::Outside,
            "empty cell ({i}, {j}, {k}) was rescued"
        );
    }
    assert!(!classes.contains(&Class::Inside));
}
