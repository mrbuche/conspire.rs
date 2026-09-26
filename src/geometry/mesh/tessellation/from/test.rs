use crate::math::assert::Assert;
use crate::{
    geometry::{
        Coordinates, Direction,
        grid::{Gradient, MarchingCubes, Voxels},
        mesh::{Connectivity, Mesh, tessellation::Tessellation, test::mesh},
    },
    math::{Tensor, assert::AssertionError},
};

pub const NORMALS: [Direction<3>; 12] = [
    Direction::const_from([0.0, 0.0, -1.0]),
    Direction::const_from([0.0, 0.0, -1.0]),
    Direction::const_from([0.0, 0.0, 1.0]),
    Direction::const_from([0.0, 0.0, 1.0]),
    Direction::const_from([0.0, -1.0, 0.0]),
    Direction::const_from([0.0, -1.0, 0.0]),
    Direction::const_from([0.0, 1.0, 0.0]),
    Direction::const_from([0.0, 1.0, 0.0]),
    Direction::const_from([-1.0, 0.0, 0.0]),
    Direction::const_from([-1.0, 0.0, 0.0]),
    Direction::const_from([1.0, 0.0, 0.0]),
    Direction::const_from([1.0, 0.0, 0.0]),
];

pub fn tessellation() -> Tessellation {
    Tessellation::from(mesh())
}

#[test]
fn isosurface_faces_outward() {
    let nel = [6, 6, 6];
    let data = (0..216)
        .map(|i| {
            let [z, y, x] = [i / 36, i / 6 % 6, i % 6];
            f64::from(u8::from(
                (2..4).contains(&z) && (2..4).contains(&y) && (2..4).contains(&x),
            ))
        })
        .collect();
    let marching = MarchingCubes {
        gradient: Gradient::Ascent,
        degenerate: false,
        ..Default::default()
    };
    let surface = marching
        .extract(&Voxels::new_row_major(data, nel), None)
        .unwrap();
    let triangles = surface.faces.len();
    let tessellation = Tessellation::from(surface);
    let coordinates = tessellation.mesh().coordinates();
    let connectivities: Vec<&[usize]> = tessellation
        .mesh()
        .connectivities()
        .iter()
        .flatten()
        .collect();
    assert_eq!(connectivities.len(), triangles);
    let normals: Vec<_> = tessellation
        .normals()
        .iter()
        .flat_map(|block| block.iter())
        .collect();
    for (triangle, normal) in connectivities.iter().zip(normals) {
        let outward: f64 = (0..3)
            .map(|axis| {
                let centroid = triangle
                    .iter()
                    .map(|&node| coordinates[node][axis].value())
                    .sum::<f64>()
                    / 3.0;
                (centroid - 2.5) * normal[axis].value()
            })
            .sum();
        assert!(outward > 0.0);
    }
}

#[test]
fn triangular_mesh() -> Result<(), AssertionError> {
    let connectivities = vec![Connectivity::Triangular(
        vec![[0_usize, 1, 2], [0, 3, 1]].into(),
    )];
    let coordinates = Coordinates::from(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ]);
    let mesh = Mesh::from((connectivities, coordinates));
    let tessellation = Tessellation::from(mesh);
    let up = Direction::const_from([0.0, 0.0, 1.0]);
    tessellation
        .normals()
        .iter()
        .flat_map(|block| block.iter())
        .try_for_each(|normal| Assert::eq(normal, &up))
}
