use crate::geometry::grid::Voxels;

#[test]
fn round_trips() {
    let mut data = vec![1u8; 64];
    data[1 + 4 + 16] = 2;
    let cleaned = Voxels::new(data, [4, 4, 4]).defeature(2);
    assert_eq!(*cleaned.nel(), [4, 4, 4]);
    assert_eq!(cleaned.data(), [1u8; 64]);
}
