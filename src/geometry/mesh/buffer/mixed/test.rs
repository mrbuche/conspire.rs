use crate::{
    geometry::mesh::{
        Connectivity, Fitting, Mesh, Verdict, buffer::test::core,
        tessellation::from::test::tessellation,
    },
    math::{Scalar, Tensor},
};
use std::collections::HashMap;

fn worst_scaled_jacobian(mesh: &Mesh<3>) -> Scalar {
    mesh.minimum_scaled_jacobians()
        .iter()
        .flatten()
        .fold(Scalar::INFINITY, |worst, &quality| worst.min(quality))
}

#[test]
fn buffer_mixed_raises_five_pyramids_per_boundary_face() {
    let faces = core().exterior_faces().len();
    let mesh = core().buffer_mixed(&tessellation(), Fitting::Soft).unwrap();
    assert_eq!(mesh.number_of_element_blocks(), 2);
    assert!(matches!(
        mesh.connectivities(),
        [Connectivity::Hexahedral(_), Connectivity::Pyramidal(_)]
    ));
    assert_eq!(mesh.number_of_elements(), 1 + 5 * faces);
    assert_eq!(mesh.coordinates().len(), 8 + 8 + faces);
    assert!(worst_scaled_jacobian(&mesh) > 0.0);
}

#[test]
fn buffer_mixed_boundary_is_manifold() {
    let mesh = core().buffer_mixed(&tessellation(), Fitting::Soft).unwrap();
    let mut edges: HashMap<[usize; 2], u32> = HashMap::new();
    for face in mesh.exterior_faces() {
        for i in 0..face.len() {
            let mut edge = [face[i], face[(i + 1) % face.len()]];
            edge.sort_unstable();
            *edges.entry(edge).or_insert(0) += 1;
        }
    }
    assert!(
        edges.values().all(|&count| count == 2),
        "non-manifold boundary: {:?}",
        edges.iter().filter(|&(_, &c)| c != 2).collect::<Vec<_>>()
    );
}

#[test]
fn buffer_mixed_captures_corners() {
    let mesh = core().buffer_mixed(&tessellation(), Fitting::Soft).unwrap();
    let coordinates = mesh.coordinates();
    let corners: std::collections::HashSet<[u8; 3]> = (8..16)
        .map(|node| {
            let point = &coordinates[node];
            point.iter().for_each(|&entry| {
                assert!(
                    (entry.value() - entry.value().round()).abs() < 0.05,
                    "duplicate node off corner: {point}"
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
}

#[test]
fn buffer_mixed_snaps_its_layer_onto_the_surface() {
    let target = tessellation();
    let mesh = core().buffer_mixed(&target, Fitting::Snap).unwrap();
    let surface = target.mesh();
    let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
    let bvh = target.bvh();
    let coordinates = mesh.coordinates();
    let deviation = (8..coordinates.len())
        .map(|node| {
            let (point, _) = bvh
                .closest_point(&coordinates[node], surface.coordinates(), &elements)
                .unwrap();
            (&coordinates[node] - point).norm().value()
        })
        .fold(0.0, Scalar::max);
    assert!(deviation < 1.0e-12, "layer deviation: {deviation}");
    assert!(worst_scaled_jacobian(&mesh) > 0.0);
}

/// The reason the shell exists: a simplicial outer surface cannot be forced
/// non-planar, so the fit has a feasible minimum where the hexahedral shell,
/// fighting a crease across every core face of the cube, does not.
#[test]
fn buffer_mixed_beats_the_hexahedral_shell_on_the_cube() {
    let hexahedral = core().buffer(&tessellation(), Fitting::Soft).unwrap();
    let mixed = core().buffer_mixed(&tessellation(), Fitting::Soft).unwrap();
    let (hex, mix) = (
        worst_scaled_jacobian(&hexahedral),
        worst_scaled_jacobian(&mixed),
    );
    assert!(mix > hex, "mixed {mix} did not beat hexahedral {hex}");
}
