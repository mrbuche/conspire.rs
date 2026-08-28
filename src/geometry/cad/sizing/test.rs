use super::FeatureSizing;
use crate::{
    geometry::{Coordinate, cad::brep::test::unit_cube},
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
    let field = FeatureSizing::of(&unit_cube(), 2, length(0.05), length(10.0), 1.0);
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
    let coarse = FeatureSizing::of(&unit_cube(), 1, length(0.01), length(10.0), 1.0);
    let fine = FeatureSizing::of(&unit_cube(), 4, length(0.01), length(10.0), 1.0);
    assert!((coarse.at(&point([0.5, 0.0, 0.0])).value() - 1.0).abs() < 1e-12);
    assert!((fine.at(&point([0.5, 0.0, 0.0])).value() - 0.25).abs() < 1e-12);
}

#[test]
fn respects_the_clamps() {
    let capped = FeatureSizing::of(&unit_cube(), 2, length(0.05), length(0.3), 1.0);
    assert!((capped.at(&point([0.5, 0.5, 0.5])).value() - 0.3).abs() < 1e-12);
    let floored = FeatureSizing::of(&unit_cube(), 8, length(0.4), length(10.0), 1.0);
    assert!((floored.at(&point([0.5, 0.0, 0.0])).value() - 0.4).abs() < 1e-12);
}

#[test]
fn obeys_the_gradation_bound() {
    let gradation = 0.7;
    let field = FeatureSizing::of(&unit_cube(), 2, length(0.05), length(10.0), gradation);
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
