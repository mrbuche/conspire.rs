use crate::{
    geometry::{
        mesh::{Tessellation, Verdict},
        ntree::{Balancing, CurvatureSizing},
    },
    math::{Scalar, Tensor, assert::AssertionError},
};
use std::path::Path;

#[test]
fn dualize_bone() -> Result<(), AssertionError> {
    let Ok(target) = Tessellation::try_from(Path::new("bone_tri.stl")) else {
        return Ok(());
    };
    let mesh = target
        .dualize(Balancing::Strong, 5.0, CurvatureSizing::default())
        .unwrap();
    let surface = target.mesh();
    let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
    let bvh = target.bvh();
    let coordinates = mesh.coordinates();
    let mut exterior: Vec<usize> = mesh.exterior_faces().into_iter().flatten().collect();
    exterior.sort_unstable();
    exterior.dedup();
    let deviation = exterior
        .iter()
        .map(|&node| {
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
    let qualities: Vec<Scalar> = mesh
        .minimum_scaled_jacobians()
        .into_iter()
        .flatten()
        .collect();
    let mean = qualities.iter().sum::<Scalar>() / qualities.len() as Scalar;
    println!(
        "elements: {}, deviation: {deviation:.6e}, worst scaled jacobian: {worst:.6}, mean {mean:.6}",
        mesh.number_of_elements()
    );
    #[cfg(feature = "netcdf")]
    crate::io::Write::write(
        &mesh,
        crate::geometry::mesh::write::Output::Exodus(Path::new("target/dualize_bone.exo")),
    )
    .unwrap();
    assert!(deviation < 4.0e-3, "deviation: {deviation}");
    assert!(worst > 0.25, "minimum scaled jacobian: {worst}");
    Ok(())
}
