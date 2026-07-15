use crate::{
    geometry::{
        Coordinates,
        bvh::BoundingVolumeHierarchy,
        mesh::{Connectivity, Mesh, tessellation::from::test::tessellation},
    },
    math::{
        Tensor,
        assert::{AssertionError, assert_eq_within_tols},
    },
};
use std::array::from_fn;

#[test]
fn buffer_adds_conforming_layer() -> Result<(), AssertionError> {
    let tessellation = tessellation();
    let bvh = BoundingVolumeHierarchy::from(&tessellation);
    let coordinates = Coordinates::from(vec![
        [0.4, 0.4, 0.1],
        [0.6, 0.4, 0.1],
        [0.6, 0.6, 0.1],
        [0.4, 0.6, 0.1],
        [0.4, 0.4, 0.2],
        [0.6, 0.4, 0.2],
        [0.6, 0.6, 0.2],
        [0.4, 0.6, 0.2],
    ]);
    let connectivities = vec![Connectivity::Hexahedral(
        vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
    )];
    let core = Mesh::from((connectivities, coordinates));
    let result = tessellation.buffer(core, &bvh).unwrap();
    let coordinates = result.coordinates();
    assert_eq!(coordinates.len(), 16);
    let hexes: Vec<[usize; 8]> = result
        .iter()
        .flatten()
        .map(|hex| from_fn(|i| hex[i]))
        .collect();
    assert_eq!(hexes.len(), 7);
    hexes[1..].iter().try_for_each(|hex| {
        (0..4).try_for_each(|k| {
            let inner = &coordinates[hex[k]];
            let projected = [inner[0], inner[1], 0.0].into();
            assert_eq_within_tols(&coordinates[hex[k + 4]], &projected)
        })
    })
}
