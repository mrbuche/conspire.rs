use crate::geometry::mesh::{Connectivity, Mesh, write::WriteExodus};

#[test]
fn two_cubes() {
    let connectivities = vec![
        Connectivity::Hexahedral(vec![[0, 1, 4, 3, 6, 7, 10, 9]].into()),
        Connectivity::Hexahedral(vec![[1, 2, 5, 4, 7, 8, 11, 10]].into()),
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

#[test]
fn two_polys() {
    let connectivities = vec![
        Connectivity::Polyhedral(
            (
                vec![vec![0, 1, 2, 3, 4, 5]],
                vec![
                    vec![0, 1, 4, 3],
                    vec![6, 7, 10, 9],
                    vec![0, 1, 7, 6],
                    vec![1, 4, 10, 7],
                    vec![4, 3, 9, 10],
                    vec![3, 0, 6, 9],
                ],
            )
                .into(),
        ),
        Connectivity::Polyhedral(
            (
                vec![vec![0, 1, 2, 3, 4, 5]],
                vec![
                    vec![1, 2, 5, 4],
                    vec![7, 8, 11, 10],
                    vec![1, 2, 8, 7],
                    vec![2, 5, 11, 8],
                    vec![5, 4, 10, 11],
                    vec![4, 1, 7, 10],
                ],
            )
                .into(),
        ),
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
    mesh.write_exodus("target/two_polys.exo").unwrap()
}
