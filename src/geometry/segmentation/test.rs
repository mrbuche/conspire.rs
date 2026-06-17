use crate::geometry::{Coordinate, grid::Grid, segmentation::Segmentation};

#[test]
fn from_grid_defaults() {
    let seg = Segmentation::from(Grid::new(vec![0u8; 6], [2, 3, 1]));
    assert_eq!(seg.scale(), &Coordinate::from([1.0, 1.0, 1.0]));
    assert_eq!(seg.translate(), &Coordinate::from([0.0, 0.0, 0.0]));
    assert_eq!(seg.nel(), &[2, 3, 1]);
}

#[test]
fn extract_shifts_origin() {
    let data: Vec<u8> = (0..24).collect();
    let seg = Segmentation::new(
        Grid::new(data, [2, 3, 4]),
        Coordinate::from([0.5, 2.0, 1.0]),
        Coordinate::from([10.0, 20.0, 30.0]),
    );
    let sub = seg.extract([0..1, 1..3, 0..2]);
    assert_eq!(sub.nel(), &[1, 2, 2]);
    assert_eq!(sub.scale(), &Coordinate::from([0.5, 2.0, 1.0]));
    assert_eq!(sub.translate(), &Coordinate::from([10.0, 22.0, 30.0]));
    assert_eq!(sub[[0, 0, 0]], seg[[0, 1, 0]]);
    assert_eq!(sub[[0, 1, 1]], seg[[0, 2, 1]]);
}

#[test]
fn diff_via_deref() {
    let a = Segmentation::from(Grid::new(vec![0u8, 1, 2, 3], [2, 2, 1]));
    let b = Segmentation::from(Grid::new(vec![0u8, 9, 2, 9], [2, 2, 1]));
    let d = a.diff(&b);
    assert_eq!(d.data(), vec![0u8, 1, 0, 1]);
}

#[test]
#[should_panic]
fn non_positive_scale_panics() {
    Segmentation::new(
        Grid::<3, u8>::new(vec![0; 1], [1, 1, 1]),
        Coordinate::from([1.0, 0.0, 1.0]),
        Coordinate::from([0.0; 3]),
    );
}
