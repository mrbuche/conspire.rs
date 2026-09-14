use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Connectivity, Fitting, Mesh, Tessellation, Verdict,
            tessellation::from::test::tessellation, test::sphere,
        },
    },
    math::{Quantity, Scalar, Tensor, assert::AssertionError},
};
use std::collections::{HashMap, HashSet};

pub(crate) fn core() -> Mesh<3> {
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
    assert_eq!(mesh.number_of_element_blocks(), 1);
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

fn trimmed_tets(spacing: Scalar) -> (Tessellation, Mesh<3>) {
    let tessellation = sphere(12, 16, 1.0);
    let (mut mesh, _) = tessellation
        .lattice_tet_background(Quantity::new(spacing))
        .unwrap();
    tessellation.trim(&mut mesh).unwrap();
    (tessellation, mesh)
}

fn deviation(mesh: &Mesh<3>, target: &Tessellation, first: usize) -> Scalar {
    let surface = target.mesh();
    let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
    let bvh = target.bvh();
    let coordinates = mesh.coordinates();
    (first..mesh.number_of_nodes())
        .map(|node| {
            let (point, _) = bvh
                .closest_point(&coordinates[node], surface.coordinates(), &elements)
                .unwrap();
            (&coordinates[node] - point).norm().value()
        })
        .fold(0.0, Scalar::max)
}

fn worst_scaled_jacobian(mesh: &Mesh<3>) -> Scalar {
    mesh.minimum_scaled_jacobians()
        .iter()
        .flatten()
        .fold(Scalar::INFINITY, |worst, &quality| worst.min(quality))
}

#[test]
fn buffer_tets_raises_three_tetrahedra_per_boundary_triangle() {
    let (tessellation, trimmed) = trimmed_tets(0.3);
    let (elements, nodes) = (trimmed.number_of_elements(), trimmed.number_of_nodes());
    let faces = trimmed.exterior_faces().len();
    let boundary: HashSet<usize> = trimmed.exterior_faces().into_iter().flatten().collect();
    let mesh = trimmed.buffer_tets(&tessellation, Fitting::Soft).unwrap();
    assert_eq!(mesh.number_of_element_blocks(), 1);
    assert!(matches!(
        mesh.connectivities(),
        [Connectivity::Tetrahedral(_)]
    ));
    assert!(mesh.number_of_elements() > elements);
    assert_eq!(mesh.number_of_elements(), elements + 3 * faces);
    assert_eq!(mesh.number_of_nodes(), nodes + boundary.len());
    assert!(worst_scaled_jacobian(&mesh) > 0.0);
    let mut counts = HashMap::<[usize; 3], usize>::new();
    mesh.iter().flatten().for_each(|tet| {
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            .iter()
            .for_each(|local| {
                let mut face = [tet[local[0]], tet[local[1]], tet[local[2]]];
                face.sort_unstable();
                *counts.entry(face).or_default() += 1
            })
    });
    assert!(counts.values().all(|&count| count <= 2));
    assert_eq!(counts.values().filter(|&&count| count == 1).count(), faces);
}

#[test]
fn buffer_tets_settles_its_layer_on_the_surface() {
    let (tessellation, trimmed) = trimmed_tets(0.3);
    let nodes = trimmed.number_of_nodes();
    let before = deviation(&trimmed, &tessellation, 0);
    let mesh = trimmed.buffer_tets(&tessellation, Fitting::Soft).unwrap();
    let after = deviation(&mesh, &tessellation, nodes);
    assert!(after < 0.25 * before, "{after} against {before}");
    assert!(worst_scaled_jacobian(&mesh) > 0.0);
}

#[test]
fn buffer_tets_snaps_its_layer_onto_the_surface() {
    let (tessellation, trimmed) = trimmed_tets(0.3);
    let nodes = trimmed.number_of_nodes();
    let mesh = trimmed.buffer_tets(&tessellation, Fitting::Snap).unwrap();
    assert!(
        deviation(&mesh, &tessellation, nodes) < 1.0e-12,
        "{}",
        deviation(&mesh, &tessellation, nodes)
    );
    assert!(worst_scaled_jacobian(&mesh) > 0.0);
}

#[test]
fn buffer_tets_rejects_a_hexahedral_mesh() {
    assert_eq!(
        core().buffer_tets(&tessellation(), Fitting::Soft).err(),
        Some("non-triangular boundary face")
    );
}

fn bar() -> Tessellation {
    let min = [0.0, 0.0, 0.0];
    let max = [3.0, 0.3, 0.3];
    let [x0, y0, z0] = min;
    let [x1, y1, z1] = max;
    let coordinates = Coordinates::from(vec![
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ]);
    let quads: [[usize; 4]; 6] = [
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
        [0, 3, 2, 1],
        [4, 5, 6, 7],
    ];
    let faces: Vec<[usize; 3]> = quads
        .iter()
        .flat_map(|&[a, b, c, d]| [[a, b, c], [a, c, d]])
        .collect();
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        coordinates,
    )))
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

#[test]
fn buffer_tets_repairs_a_pinched_trim() -> Result<(), AssertionError> {
    let target = bar();
    let (mut mesh, _) = target.lattice_tet_background(Quantity::new(0.1)).unwrap();
    target.trim(&mut mesh).unwrap();
    let mesh = mesh.buffer_tets(&target, Fitting::Soft).unwrap();
    assert_eq!(mesh.number_of_element_blocks(), 1);
    assert_manifold_boundary(&mesh);
    assert!(worst_scaled_jacobian(&mesh) > 0.0);
    Ok(())
}
