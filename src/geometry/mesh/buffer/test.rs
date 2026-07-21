use crate::{
    geometry::{
        Coordinate, Coordinates,
        grid::Voxels,
        mesh::{Connectivity, Mesh, Tessellation, Verdict, tessellation::from::test::tessellation},
        segmentation::Segmentation,
    },
    math::{Scalar, Tensor, assert::AssertionError},
};
use std::{collections::HashSet, path::Path};

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
    let mesh = core().buffer(&tessellation()).unwrap();
    assert_eq!(mesh.coordinates().len(), 16);
    assert_eq!(mesh.number_of_element_blocks(), 2);
    assert_eq!(mesh.iter().flatten().count(), 7);
    let coordinates = mesh.coordinates();
    let corners: HashSet<[u8; 3]> = (8..16)
        .map(|node| {
            let point = &coordinates[node];
            point.iter().for_each(|&entry| {
                assert!(
                    (entry - entry.round()).abs() < 0.05,
                    "layer node off corner: {point}"
                )
            });
            [
                point[0].round() as u8,
                point[1].round() as u8,
                point[2].round() as u8,
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

const SIZE: Scalar = 0.02;

#[test]
fn buffer_bone_baseline() -> Result<(), AssertionError> {
    let Ok(target) = Tessellation::try_from(Path::new("bone_tri.stl")) else {
        return Ok(());
    };
    let mut translate = Coordinate::const_from([Scalar::INFINITY; 3]);
    target
        .mesh()
        .coordinates()
        .iter()
        .for_each(|point| (0..3).for_each(|ax| translate[ax] = translate[ax].min(point[ax])));
    let voxels = Voxels::from_tessellation(&target, SIZE);
    let mesh = Mesh::from_segmentation(
        Segmentation::new(voxels, Coordinate::const_from([SIZE; 3]), translate),
        Some(&[0]),
    );
    let inner = mesh.coordinates().len();
    let mesh = mesh.buffer(&target).unwrap();
    let coordinates = mesh.coordinates();
    let surface = target.mesh();
    let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
    let bvh = target.bvh();
    let deviation = (inner..coordinates.len())
        .map(|node| {
            let (point, _) = bvh
                .closest_point(&coordinates[node], surface.coordinates(), &elements)
                .unwrap();
            (&coordinates[node] - point).norm()
        })
        .fold(0.0, Scalar::max);
    let worst = mesh
        .minimum_scaled_jacobians()
        .iter()
        .flatten()
        .fold(Scalar::INFINITY, |worst, &quality| worst.min(quality));
    println!("deviation: {deviation:.6e}, worst scaled jacobian: {worst:.6}");
    #[cfg(feature = "netcdf")]
    crate::io::Write::write(
        &mesh,
        crate::geometry::mesh::write::Output::Exodus(Path::new("target/buffer_bone.exo")),
    )
    .unwrap();
    assert!(deviation < 0.25 * SIZE, "layer deviation: {deviation}");
    assert!(worst > 0.25, "minimum scaled jacobian: {worst}");
    Ok(())
}
