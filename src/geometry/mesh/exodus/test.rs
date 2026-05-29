use crate::geometry::mesh::{Connectivity, Mesh, PrimitiveConnectivity, exodus::WriteExodus};

#[test]
fn two_cubes() {
    let connectivities = vec![
        Connectivity::<i32>::Hexahedral(PrimitiveConnectivity(vec![[0, 1, 4, 3, 6, 7, 10, 9]])),
        Connectivity::<i32>::Hexahedral(PrimitiveConnectivity(vec![[1, 2, 5, 4, 7, 8, 11, 10]])),
    ];
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
    ]
    .into();
    let mesh = Mesh {
        connectivities,
        coordinates,
    };
    mesh.write_exodus("target/two_cubes.exo").unwrap()
}
