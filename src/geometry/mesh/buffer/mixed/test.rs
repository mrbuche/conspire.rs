use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{
            Connectivity, Fitting, Mesh, Tessellation, Verdict, buffer::test::core,
            tessellation::from::test::tessellation,
        },
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

fn assert_manifold_boundary(mesh: &Mesh<3>) {
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

/// A closed triangular prism: a ridge crease at `z = 1` running the length of
/// `x`, two roof slants down to `z = 0`, a floor, and two end caps.
fn ridge_prism() -> Tessellation {
    let coordinates = Coordinates::from(vec![
        [0.0, -1.5, 0.0],
        [4.0, -1.5, 0.0],
        [4.0, 1.5, 0.0],
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 1.0],
        [4.0, 0.0, 1.0],
    ]);
    let triangles: Vec<[usize; 3]> = vec![
        [0, 2, 1],
        [0, 3, 2],
        [2, 3, 4],
        [2, 4, 5],
        [0, 1, 4],
        [1, 5, 4],
        [0, 4, 3],
        [1, 2, 5],
    ];
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(triangles.into())],
        coordinates,
    )))
}

/// A flat hexahedral slab under the prism ridge: its top face sits within a
/// face size of the crease, every other face well clear of one.
fn ridge_core() -> Mesh<3> {
    let coordinates = Coordinates::from(vec![
        [1.75, -0.25, 0.42],
        [2.25, -0.25, 0.42],
        [2.25, 0.25, 0.42],
        [1.75, 0.25, 0.42],
        [1.75, -0.25, 0.57],
        [2.25, -0.25, 0.57],
        [2.25, 0.25, 0.57],
        [1.75, 0.25, 0.57],
    ]);
    Mesh::from((
        vec![Connectivity::Hexahedral(
            vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
        )],
        coordinates,
    ))
}

#[test]
fn buffer_mixed_is_all_hexahedral_when_no_face_meets_a_feature() {
    let mesh = core().buffer_mixed(&tessellation(), Fitting::Soft).unwrap();
    assert_eq!(mesh.number_of_element_blocks(), 1);
    assert!(matches!(
        mesh.connectivities(),
        [Connectivity::Hexahedral(_)]
    ));
    assert_eq!(mesh.number_of_elements(), 1 + core().exterior_faces().len());
    assert_eq!(mesh.coordinates().len(), 16);
    assert!(worst_scaled_jacobian(&mesh) > 0.2);
}

#[test]
fn buffer_mixed_raises_a_pyramid_fan_where_a_crease_crosses_a_face() {
    let mesh = ridge_core()
        .buffer_mixed(&ridge_prism(), Fitting::Soft)
        .unwrap();
    assert_eq!(mesh.number_of_element_blocks(), 2);
    assert!(matches!(
        mesh.connectivities(),
        [Connectivity::Hexahedral(_), Connectivity::Pyramidal(_)]
    ));
    let [
        Connectivity::Hexahedral(hexes),
        Connectivity::Pyramidal(pyramids),
    ] = mesh.connectivities()
    else {
        unreachable!()
    };
    assert_eq!(
        pyramids.iter().count(),
        5,
        "exactly one boundary face crossed"
    );
    assert_eq!(
        hexes.iter().count(),
        1 + 5,
        "core plus five clean shell hexes"
    );
    assert_eq!(mesh.coordinates().len(), 8 + 8 + 1);
    assert_manifold_boundary(&mesh);
    assert!(worst_scaled_jacobian(&mesh) > 0.0);
}

#[test]
fn buffer_mixed_boundary_is_manifold() {
    let mesh = core().buffer_mixed(&tessellation(), Fitting::Soft).unwrap();
    assert_manifold_boundary(&mesh);
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
    assert!(worst_scaled_jacobian(&mesh) > 0.2);
}

/// Distance from a point to the prism ridge line `{(x, 0, 1)}`.
fn ridge_distance(point: &Coordinate<3>) -> Scalar {
    (point[1].value().powi(2) + (point[2].value() - 1.0).powi(2)).sqrt()
}

/// The reason the pyramid fan exists: its apex is free to sit on the ridge, so
/// the crease is meshed at scaled Jacobian ~0.4, where the single hexahedron
/// `buffer` raises there -- four outer nodes forced to stay a planar
/// quadrilateral astride the crease -- manages only ~0.2.
///
/// The comparison is against that one straddling cell, not the whole mesh: the
/// worst element of either result is an unrelated shell hexahedron over the
/// core's floor-facing quadrilateral, which the two results share.
#[test]
fn buffer_mixed_beats_the_hexahedral_shell_across_the_crease() {
    let hexahedral = ridge_core().buffer(&ridge_prism(), Fitting::Soft).unwrap();
    let mixed = ridge_core()
        .buffer_mixed(&ridge_prism(), Fitting::Soft)
        .unwrap();
    let centroids = hexahedral.centroids();
    let straddling = hexahedral
        .minimum_scaled_jacobians()
        .iter()
        .flatten()
        .enumerate()
        .skip(1)
        .min_by(|&(a, _), &(b, _)| {
            ridge_distance(&centroids[a]).total_cmp(&ridge_distance(&centroids[b]))
        })
        .map(|(_, &quality)| quality)
        .unwrap();
    let [_, Connectivity::Pyramidal(_)] = mixed.connectivities() else {
        unreachable!("the crease face raises a pyramid fan")
    };
    let fan = mixed.minimum_scaled_jacobians()[1]
        .iter()
        .fold(Scalar::INFINITY, |worst, &quality| worst.min(quality));
    assert!(
        fan > straddling * 1.5,
        "pyramid fan {fan} did not clearly beat the straddling hexahedron {straddling}"
    );
}
