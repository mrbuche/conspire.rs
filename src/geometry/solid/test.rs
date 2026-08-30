use super::{SolidOracle, classify_by_flood_fill, classify_by_signed_distance};
use crate::{
    geometry::{
        Coordinate, Coordinates, Direction,
        mesh::{Class, Connectivity, Mesh},
    },
    math::Scalar,
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
        if self.lie && deep { -distance } else { distance }
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
