use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Fitting, Mesh, Verdict, tessellation::from::test::tessellation},
    },
    math::{Scalar, Tensor, assert::AssertionError},
};
use std::collections::HashSet;

fn core() -> Mesh<3> {
    let coordinates = Coordinates::from(vec![
        [0.4, 0.4, 0.4],
        [0.6, 0.4, 0.4],
        [0.6, 0.6, 0.4],
        [0.4, 0.6, 0.4],
        [0.4, 0.4, 0.6],
        [0.6, 0.4, 0.6],
        [0.6, 0.6, 0.6],
        [0.4, 0.6, 0.6],
    ]);
    let connectivities = vec![Connectivity::Hexahedral(
        vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
    )];
    Mesh::from((connectivities, coordinates))
}

#[test]
fn buffer_captures_corners() -> Result<(), AssertionError> {
    let mesh = core().buffer(&tessellation(), Fitting::Soft).unwrap();
    assert_eq!(mesh.coordinates().len(), 16);
    assert_eq!(mesh.number_of_element_blocks(), 2);
    assert_eq!(mesh.iter().flatten().count(), 7);
    let coordinates = mesh.coordinates();
    let corners: HashSet<[u8; 3]> = (8..16)
        .map(|node| {
            let point = &coordinates[node];
            point.iter().for_each(|&entry| {
                assert!(
                    (entry.value() - entry.value().round()).abs() < 0.05,
                    "layer node off corner: {point}"
                )
            });
            [
                point[0].value().round() as u8,
                point[1].value().round() as u8,
                point[2].value().round() as u8,
            ]
        })
        .collect();
    assert_eq!(corners.len(), 8);
    let worst = mesh
        .minimum_scaled_jacobians()
        .iter()
        .flatten()
        .fold(Scalar::INFINITY, |worst, &quality| worst.min(quality));
    assert!(worst > 0.2, "minimum scaled jacobian: {worst}");
    Ok(())
}

#[test]
fn buffer_snaps_to_surface() -> Result<(), AssertionError> {
    let target = tessellation();
    let mesh = core().buffer(&target, Fitting::Snap).unwrap();
    let surface = target.mesh();
    let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
    let bvh = target.bvh();
    let coordinates = mesh.coordinates();
    let deviation = (8..16)
        .map(|node| {
            let (point, _) = bvh
                .closest_point(&coordinates[node], surface.coordinates(), &elements)
                .unwrap();
            (&coordinates[node] - point).norm().value()
        })
        .fold(0.0, Scalar::max);
    assert!(deviation < 1.0e-12, "layer deviation: {deviation}");
    let worst = mesh
        .minimum_scaled_jacobians()
        .iter()
        .flatten()
        .fold(Scalar::INFINITY, |worst, &quality| worst.min(quality));
    assert!(worst > 0.2, "minimum scaled jacobian: {worst}");
    Ok(())
}
