use super::super::{
    SERIAL,
    energy::{node_scales, scales},
    test::{cube, grid},
};
use super::Surface;
use crate::{
    geometry::{Coordinates, mesh::Connectivity},
    math::{
        Quantity, QuantityVector, Scalar, Tensor,
        assert::{Assert, AssertionError},
    },
    units::Length,
};

#[test]
fn planes_do_not_depend_on_how_the_nodes_are_shared_among_threads() -> Result<(), AssertionError> {
    let mesh = grid(15, 0.5);
    let hexes: Vec<[usize; 8]> = mesh
        .iter()
        .flat_map(|block| match block {
            Connectivity::Hexahedral(elements) => elements.iter().copied().collect::<Vec<_>>(),
            _ => unreachable!(),
        })
        .collect();
    let mut boundary: Vec<usize> = mesh.exterior_faces().into_iter().flatten().collect();
    boundary.sort_unstable();
    boundary.dedup();
    assert!(boundary.len() > 2 * SERIAL);
    let coordinates = mesh.coordinates().clone();
    let sizes = node_scales(
        &boundary,
        &hexes,
        &scales(&hexes, &vec![0u8; hexes.len()], &coordinates),
        coordinates.len(),
    );
    let target = cube();
    let surface = Surface::new(&target);
    let planes = surface.planes(&boundary, &sizes, &coordinates);
    let mut expected = Vec::new();
    for (index, &node) in boundary.iter().enumerate() {
        for mut plane in surface.planes(&[node], &sizes, &coordinates) {
            plane.boundary = index;
            expected.push(plane)
        }
    }
    assert!(!planes.is_empty());
    assert_eq!(planes.len(), expected.len());
    for (plane, expected) in planes.iter().zip(&expected) {
        assert_eq!(
            (plane.boundary, plane.slot),
            (expected.boundary, expected.slot)
        );
        Assert::eq(plane.offset.value(), expected.offset.value())?;
        Assert::eq(&plane.normal, &expected.normal)?
    }
    Ok(())
}

#[test]
fn a_node_beside_an_edge_keeps_the_planes_of_both_faces() -> Result<(), AssertionError> {
    let target = cube();
    let surface = Surface::new(&target);
    let coordinates = Coordinates::from(vec![[1.005, -2.0e-5, 0.5]]);
    let sizes: QuantityVector<Length> = [Quantity::new(0.33)].into_iter().collect();
    let planes = surface.planes(&[0], &sizes, &coordinates);
    assert_eq!(planes.len(), 2);
    let has = |normal: [Scalar; 3], offset: Scalar| {
        planes.iter().any(|plane| {
            (0..3).all(|i| (plane.normal[i].value() - normal[i]).abs() < 1.0e-12)
                && (plane.offset.value() - offset).abs() < 1.0e-12
        })
    };
    assert!(has([1.0, 0.0, 0.0], 1.0));
    assert!(has([0.0, -1.0, 0.0], 0.0));
    Ok(())
}
