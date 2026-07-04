use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Connectivities, Connectivity, Input, Mesh, NodeSets, Output, SideSets,
            test::{CONNECTIVITY, COORDINATES, mesh},
        },
    },
    io::Write,
    math::Set,
};

#[test]
fn round_trip() {
    let original = mesh();
    original
        .write(Output::Exodus("target/read_exodus_round_trip.exo"))
        .unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus("target/read_exodus_round_trip.exo")).unwrap();
    let expected_coords = Coordinates::from(COORDINATES);
    assert_eq!(read.coordinates(), &expected_coords);
    match &read.connectivities()[0] {
        Connectivity::Triangular(triangles) => {
            assert!(triangles.iter().eq(CONNECTIVITY.iter()))
        }
        _ => panic!("expected Triangular block"),
    }
}

#[test]
fn round_trip_polyhedral() {
    let elements_faces = vec![vec![0_usize, 1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10, 11]];
    let faces_nodes = vec![
        vec![0_usize, 1, 4, 3],
        vec![6, 7, 10, 9],
        vec![0, 1, 7, 6],
        vec![1, 4, 10, 7],
        vec![4, 3, 9, 10],
        vec![3, 0, 6, 9],
        vec![1, 2, 5, 4],
        vec![7, 8, 11, 10],
        vec![1, 2, 8, 7],
        vec![2, 5, 11, 8],
        vec![5, 4, 10, 11],
        vec![4, 1, 7, 10],
    ];
    let connectivities = vec![Connectivity::Polyhedral(
        (elements_faces.clone(), faces_nodes.clone()).into(),
    )];
    let coordinates: Coordinates<3> = vec![
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
    let original = Mesh::from((connectivities, coordinates.clone()));
    original
        .write(Output::Exodus(
            "target/read_exodus_round_trip_polyhedral.exo",
        ))
        .unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus(
        "target/read_exodus_round_trip_polyhedral.exo",
    ))
    .unwrap();
    assert_eq!(read.coordinates(), &coordinates);
    match &read.connectivities()[0] {
        Connectivity::Polyhedral(poly) => {
            assert!(poly.iter().eq(elements_faces.iter()));
        }
        _ => panic!("expected Polyhedral block"),
    }
}

#[test]
fn round_trip_block_numbers() {
    let connectivities = vec![
        Connectivity::Triangular(vec![[0, 1, 2]].into()),
        Connectivity::Triangular(vec![[3, 4, 5]].into()),
    ];
    let coordinates: Coordinates<3> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    .into();
    let original = Mesh::from((
        Connectivities::from((connectivities, vec![10, 20])),
        coordinates.into(),
    ));
    original
        .write(Output::Exodus("target/read_exodus_block_numbers.exo"))
        .unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus("target/read_exodus_block_numbers.exo")).unwrap();
    assert_eq!(read.connectivities.numbers(), Some([10, 20].as_slice()));
}

#[test]
fn round_trip_element_numbers() {
    let mut block_0 = Connectivity::Triangular(vec![[0, 1, 2]].into());
    block_0.number_elements(vec![100]);
    let mut block_1 = Connectivity::Triangular(vec![[3, 4, 5]].into());
    block_1.number_elements(vec![200]);
    let coordinates: Coordinates<3> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    .into();
    let original = Mesh::from((vec![block_0, block_1], coordinates));
    original
        .write(Output::Exodus("target/read_exodus_element_numbers.exo"))
        .unwrap();
    let read =
        Mesh::<3>::try_from(Input::Exodus("target/read_exodus_element_numbers.exo")).unwrap();
    assert_eq!(
        read.connectivities()[0].element_numbers(),
        Some([100].as_slice())
    );
    assert_eq!(
        read.connectivities()[1].element_numbers(),
        Some([200].as_slice())
    );
}

#[test]
fn round_trip_node_numbers() {
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2]].into())];
    let coordinates: Coordinates<3> =
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]].into();
    let original = Mesh::from((
        Connectivities::from(connectivities),
        Set::from((coordinates, vec![7, 8, 9])),
    ));
    original
        .write(Output::Exodus("target/read_exodus_node_numbers.exo"))
        .unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus("target/read_exodus_node_numbers.exo")).unwrap();
    assert_eq!(read.coordinates.numbers(), Some([7, 8, 9].as_slice()));
}

#[test]
fn round_trip_node_sets() {
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2], [1, 2, 3]].into())];
    let coordinates: Coordinates<3> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    .into();
    let mut original = Mesh::from((connectivities, coordinates));
    original.set_node_sets(vec![vec![0, 1], vec![2, 3]].into());
    original
        .write(Output::Exodus("target/read_exodus_node_sets.exo"))
        .unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus("target/read_exodus_node_sets.exo")).unwrap();
    assert_eq!(read.node_sets(), &[vec![0, 1], vec![2, 3]]);
    assert_eq!(read.node_set_numbers(), Some([1, 2].as_slice()));
}

#[test]
fn round_trip_node_set_numbers() {
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2], [1, 2, 3]].into())];
    let coordinates: Coordinates<3> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    .into();
    let mut original = Mesh::from((connectivities, coordinates));
    original.set_node_sets(NodeSets::from((vec![vec![0, 1], vec![2, 3]], vec![10, 20])));
    original
        .write(Output::Exodus("target/read_exodus_node_set_numbers.exo"))
        .unwrap();
    let read =
        Mesh::<3>::try_from(Input::Exodus("target/read_exodus_node_set_numbers.exo")).unwrap();
    assert_eq!(read.node_sets(), &[vec![0, 1], vec![2, 3]]);
    assert_eq!(read.node_set_numbers(), Some([10, 20].as_slice()));
}

#[test]
fn round_trip_side_sets() {
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2], [1, 2, 3]].into())];
    let coordinates: Coordinates<3> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    .into();
    let mut original = Mesh::from((connectivities, coordinates));
    original.set_side_sets(vec![vec![(0, 1)], vec![(0, 2), (1, 0)]].into());
    original
        .write(Output::Exodus("target/read_exodus_side_sets.exo"))
        .unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus("target/read_exodus_side_sets.exo")).unwrap();
    assert_eq!(read.side_sets(), &[vec![(0, 1)], vec![(0, 2), (1, 0)]]);
    assert_eq!(read.side_set_numbers(), Some([1, 2].as_slice()));
}

#[test]
fn round_trip_side_set_numbers() {
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2], [1, 2, 3]].into())];
    let coordinates: Coordinates<3> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    .into();
    let mut original = Mesh::from((connectivities, coordinates));
    original.set_side_sets(SideSets::from((
        vec![vec![(0, 1)], vec![(1, 0)]],
        vec![10, 20],
    )));
    original
        .write(Output::Exodus("target/read_exodus_side_set_numbers.exo"))
        .unwrap();
    let read =
        Mesh::<3>::try_from(Input::Exodus("target/read_exodus_side_set_numbers.exo")).unwrap();
    assert_eq!(read.side_sets(), &[vec![(0, 1)], vec![(1, 0)]]);
    assert_eq!(read.side_set_numbers(), Some([10, 20].as_slice()));
}

#[test]
fn round_trip_side_sets_with_custom_element_numbers() {
    let mut block_0 = Connectivity::Triangular(vec![[0, 1, 2]].into());
    block_0.number_elements(vec![100]);
    let mut block_1 = Connectivity::Triangular(vec![[1, 2, 3]].into());
    block_1.number_elements(vec![200]);
    let coordinates: Coordinates<3> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    .into();
    let mut original = Mesh::from((vec![block_0, block_1], coordinates));
    original.set_side_sets(vec![vec![(0, 0), (1, 2)]].into());
    original
        .write(Output::Exodus(
            "target/read_exodus_side_sets_custom_elements.exo",
        ))
        .unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus(
        "target/read_exodus_side_sets_custom_elements.exo",
    ))
    .unwrap();
    assert_eq!(read.side_sets(), &[vec![(0, 0), (1, 2)]]);
}
