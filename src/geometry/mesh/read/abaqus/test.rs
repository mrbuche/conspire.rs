use super::ReadAbaqus;
use crate::{
    geometry::mesh::{Connectivity, Mesh, Output},
    io::Write,
};
use std::fs::write;

fn first_element(mesh: &Mesh<3>, block: usize) -> &[usize] {
    mesh.iter().nth(block).unwrap().iter().next().unwrap()
}

#[test]
fn round_trip_mixed() {
    let connectivities = vec![
        Connectivity::Hexahedral(vec![[0, 1, 2, 3, 4, 5, 6, 7]].into()),
        Connectivity::Wedge(vec![[4, 5, 8, 7, 6, 9]].into()),
        Connectivity::Pyramidal(vec![[1, 2, 6, 5, 10]].into()),
        Connectivity::Tetrahedral(vec![[1, 2, 10, 11]].into()),
    ];
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.0, 2.0],
        [0.5, 1.0, 2.0],
        [2.0, 0.5, 0.5],
        [1.5, 0.5, -1.0],
    ]
    .into();
    let path = "target/round_trip.inp";
    Mesh::from((connectivities, coordinates))
        .write(Output::Abaqus(path))
        .unwrap();
    let mesh = Mesh::<3>::read_abaqus(path).unwrap();
    assert_eq!(mesh.number_of_nodes(), 12);
    assert_eq!(mesh.number_of_element_blocks(), 4);
    assert_eq!(mesh.number_of_elements(), 4);
    assert_eq!(first_element(&mesh, 0), [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(first_element(&mesh, 1), [4, 5, 8, 7, 6, 9]);
    assert_eq!(first_element(&mesh, 2), [1, 2, 6, 5, 10]);
    assert_eq!(first_element(&mesh, 3), [1, 2, 10, 11]);
    let coordinates = mesh.coordinates();
    assert_eq!(
        [coordinates[10][0], coordinates[10][1], coordinates[10][2]],
        [2.0, 0.5, 0.5]
    );
}

#[test]
fn reads_sparse_ids_continuation_and_comments() {
    let path = "target/sparse.inp";
    write(
        path,
        "*Heading\n deck\n*Node\n\
         100, 0., 0., 0.\n200, 1., 0., 0.\n300, 1., 1., 0.\n400, 0., 1., 0.\n\
         500, 0., 0., 1.\n600, 1., 0., 1.\n700, 1., 1., 1.\n800, 0., 1., 1.\n\
         ** a comment between data\n\
         *Element, TYPE=C3D8, ELSET=Brick\n\
         5, 100, 200, 300, 400,\n   500, 600, 700, 800\n",
    )
    .unwrap();
    let mesh = Mesh::<3>::read_abaqus(path).unwrap();
    assert_eq!(mesh.number_of_nodes(), 8);
    assert_eq!(mesh.number_of_element_blocks(), 1);
    assert_eq!(first_element(&mesh, 0), [0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn unknown_element_type_errors() {
    let path = "target/bad.inp";
    write(path, "*Node\n1, 0., 0., 0.\n*Element, type=C3D20\n1, 1\n").unwrap();
    assert!(Mesh::<3>::read_abaqus(path).is_err());
}

#[test]
fn round_trip_node_sets() {
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2], [1, 2, 3]].into())];
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    .into();
    let mut mesh = Mesh::from((connectivities, coordinates));
    mesh.set_node_sets(vec![vec![0, 1], vec![2, 3]].into());
    let path = "target/round_trip_node_sets.inp";
    mesh.write(Output::Abaqus(path)).unwrap();
    let read = Mesh::<3>::read_abaqus(path).unwrap();
    assert_eq!(read.node_sets(), &[vec![0, 1], vec![2, 3]]);
}

#[test]
fn reads_nset_with_sparse_ids() {
    let path = "target/nset_sparse.inp";
    write(
        path,
        "*Heading\n deck\n*Node\n\
         100, 0., 0., 0.\n200, 1., 0., 0.\n300, 1., 1., 0.\n400, 0., 1., 0.\n\
         *Element, TYPE=S4, ELSET=Quad\n1, 100, 200, 300, 400\n\
         *Nset, nset=FIXED\n100, 200,\n300\n",
    )
    .unwrap();
    let mesh = Mesh::<3>::read_abaqus(path).unwrap();
    assert_eq!(mesh.node_sets(), &[vec![0, 1, 2]]);
}

fn round_trips_all_faces(connectivity: Connectivity, coordinates: Vec<[f64; 3]>, path: &str) {
    let num_faces = connectivity.local_faces().len();
    let mut mesh = Mesh::from((vec![connectivity], coordinates.into()));
    let sides: Vec<(usize, usize)> = (0..num_faces).map(|ordinal| (0, ordinal)).collect();
    mesh.set_side_sets(vec![sides.clone()].into());
    mesh.write(Output::Abaqus(path)).unwrap();
    let read = Mesh::<3>::read_abaqus(path).unwrap();
    assert_eq!(read.side_sets(), &[sides]);
}

#[test]
fn round_trip_side_sets_hexahedral() {
    round_trips_all_faces(
        Connectivity::Hexahedral(vec![[0, 1, 2, 3, 4, 5, 6, 7]].into()),
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        "target/round_trip_side_sets_hex.inp",
    );
}

#[test]
fn round_trip_side_sets_wedge() {
    round_trips_all_faces(
        Connectivity::Wedge(vec![[0, 1, 2, 3, 4, 5]].into()),
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        "target/round_trip_side_sets_wedge.inp",
    );
}

#[test]
fn round_trip_side_sets_pyramidal() {
    round_trips_all_faces(
        Connectivity::Pyramidal(vec![[0, 1, 2, 3, 4]].into()),
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ],
        "target/round_trip_side_sets_pyramid.inp",
    );
}

#[test]
fn round_trip_side_sets_tetrahedral() {
    round_trips_all_faces(
        Connectivity::Tetrahedral(vec![[0, 1, 2, 3]].into()),
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "target/round_trip_side_sets_tet.inp",
    );
}

#[test]
fn round_trip_side_sets_quadrilateral() {
    round_trips_all_faces(
        Connectivity::Quadrilateral(vec![[0, 1, 2, 3]].into()),
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        "target/round_trip_side_sets_quad.inp",
    );
}

#[test]
fn round_trip_side_sets_triangular() {
    round_trips_all_faces(
        Connectivity::Triangular(vec![[0, 1, 2]].into()),
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "target/round_trip_side_sets_tri.inp",
    );
}

#[test]
fn reads_surface_with_sparse_element_ids() {
    let path = "target/surface_sparse.inp";
    write(
        path,
        "*Heading\n deck\n*Node\n\
         1, 0., 0., 0.\n2, 1., 0., 0.\n3, 1., 1., 0.\n4, 0., 1., 0.\n\
         *Element, TYPE=S4, ELSET=Quad\n5, 1, 2, 3, 4\n\
         *Surface, type=ELEMENT, name=TOP\n5, S2\n",
    )
    .unwrap();
    let mesh = Mesh::<3>::read_abaqus(path).unwrap();
    assert_eq!(mesh.side_sets(), &[vec![(0, 1)]]);
}
