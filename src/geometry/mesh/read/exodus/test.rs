use super::face_offset;
use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Connectivities, Connectivity, ExodusFormat, Input, Mesh, NodeSets, Output, SideSets,
            test::{CONNECTIVITY, COORDINATES, mesh},
        },
    },
    io::{GetVariable, NetCDF, Write},
};
use std::array::from_fn;

#[test]
fn round_trip() {
    let original = mesh();
    original
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_round_trip.exo",
        )))
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
    let coordinates = Coordinates::from(vec![
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
    ]);
    let original = Mesh::from((connectivities, coordinates.clone()));
    original
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_round_trip_polyhedral.exo",
        )))
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
fn round_trip_block_labels() {
    let connectivities = vec![
        Connectivity::Triangular(vec![[0, 1, 2]].into()),
        Connectivity::Triangular(vec![[3, 4, 5]].into()),
    ];
    let coordinates = Coordinates::from(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]);
    let original = Mesh::from((
        Connectivities::from((connectivities, vec![10, 20])),
        coordinates,
    ));
    original
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_block_labels.exo",
        )))
        .unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus("target/read_exodus_block_labels.exo")).unwrap();
    assert_eq!(read.connectivities.labels(), Some([10, 20].as_slice()));
}

#[test]
fn round_trip_element_labels() {
    let mut block_0 = Connectivity::Triangular(vec![[0, 1, 2]].into());
    block_0.label_elements(vec![100]);
    let mut block_1 = Connectivity::Triangular(vec![[3, 4, 5]].into());
    block_1.label_elements(vec![200]);
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
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_element_labels.exo",
        )))
        .unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus("target/read_exodus_element_labels.exo")).unwrap();
    assert_eq!(
        read.connectivities()[0].element_labels(),
        Some([100].as_slice())
    );
    assert_eq!(
        read.connectivities()[1].element_labels(),
        Some([200].as_slice())
    );
}

#[test]
fn round_trip_node_labels() {
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2]].into())];
    let coordinates: Coordinates<3> =
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]].into();
    let original = Mesh::from((
        Connectivities::from(connectivities),
        coordinates,
        vec![7, 8, 9],
    ));
    original
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_node_labels.exo",
        )))
        .unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus("target/read_exodus_node_labels.exo")).unwrap();
    assert_eq!(read.node_labels(), Some([7, 8, 9].as_slice()));
}

#[test]
fn round_trip_without_node_labels() {
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2]].into())];
    let coordinates: Coordinates<3> =
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]].into();
    Mesh::from((connectivities, coordinates))
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_without_node_labels.exo",
        )))
        .unwrap();
    let read =
        Mesh::<3>::try_from(Input::Exodus("target/read_exodus_without_node_labels.exo")).unwrap();
    assert!(read.node_labels().is_none());
}

#[test]
fn round_trip_node_sets() {
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2], [1, 2, 3]].into())];
    let coordinates = Coordinates::from(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]);
    let mut original = Mesh::from((connectivities, coordinates));
    original.set_node_sets(vec![vec![0, 1], vec![2, 3]].into());
    original
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_node_sets.exo",
        )))
        .unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus("target/read_exodus_node_sets.exo")).unwrap();
    assert_eq!(
        read.node_sets(),
        &NodeSets::from((vec![vec![0, 1], vec![2, 3]], vec![1, 2]))
    );
}

#[test]
fn round_trip_node_set_labels() {
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2], [1, 2, 3]].into())];
    let coordinates = Coordinates::from(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]);
    let mut original = Mesh::from((connectivities, coordinates));
    original.set_node_sets(NodeSets::from((vec![vec![0, 1], vec![2, 3]], vec![10, 20])));
    original
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_node_set_labels.exo",
        )))
        .unwrap();
    let read =
        Mesh::<3>::try_from(Input::Exodus("target/read_exodus_node_set_labels.exo")).unwrap();
    assert_eq!(
        read.node_sets(),
        &NodeSets::from((vec![vec![0, 1], vec![2, 3]], vec![10, 20]))
    );
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
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_side_sets.exo",
        )))
        .unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus("target/read_exodus_side_sets.exo")).unwrap();
    assert_eq!(
        read.side_sets(),
        &SideSets::from((vec![vec![(0, 1)], vec![(0, 2), (1, 0)]], vec![1, 2]))
    );
}

#[test]
fn round_trip_side_set_labels() {
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
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_side_set_labels.exo",
        )))
        .unwrap();
    let read =
        Mesh::<3>::try_from(Input::Exodus("target/read_exodus_side_set_labels.exo")).unwrap();
    assert_eq!(
        read.side_sets(),
        &SideSets::from((vec![vec![(0, 1)], vec![(1, 0)]], vec![10, 20]))
    );
}

#[test]
fn round_trip_side_sets_with_custom_element_labels() {
    let mut block_0 = Connectivity::Triangular(vec![[0, 1, 2]].into());
    block_0.label_elements(vec![100]);
    let mut block_1 = Connectivity::Triangular(vec![[1, 2, 3]].into());
    block_1.label_elements(vec![200]);
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
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_side_sets_custom_elements.exo",
        )))
        .unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus(
        "target/read_exodus_side_sets_custom_elements.exo",
    ))
    .unwrap();
    assert_eq!(
        read.side_sets(),
        &SideSets::from((vec![vec![(0, 0), (1, 2)]], vec![1]))
    );
}

#[test]
fn round_trip_polygonal() {
    let elements_faces = vec![vec![0_usize, 1, 2, 3]];
    let faces_nodes = vec![vec![0_usize, 1], vec![1, 2], vec![2, 3], vec![3, 0]];
    let connectivities = vec![Connectivity::Polygonal(
        (elements_faces.clone(), faces_nodes).into(),
    )];
    let coordinates: Coordinates<3> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    .into();
    let original = Mesh::from((connectivities, coordinates.clone()));
    original
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_round_trip_polygonal.exo",
        )))
        .unwrap();
    let read =
        Mesh::<3>::try_from(Input::Exodus("target/read_exodus_round_trip_polygonal.exo")).unwrap();
    assert_eq!(read.coordinates(), &coordinates);
    match &read.connectivities()[0] {
        Connectivity::Polygonal(poly) => {
            assert!(poly.iter().eq(elements_faces.iter()));
        }
        _ => panic!("expected Polygonal block"),
    }
}

const FIXTURES: &str = "src/geometry/mesh/read/exodus/fixtures";

fn cube(nodes: [usize; 8]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    (
        vec![vec![0, 1, 2, 3, 4, 5]],
        [
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
            [0, 3, 2, 1],
            [4, 5, 6, 7],
        ]
        .iter()
        .map(|face| face.iter().map(|&node| nodes[node]).collect())
        .collect(),
    )
}

fn cube_nodes(first: usize) -> [usize; 8] {
    from_fn(|node| first + node)
}

fn mixed_coordinates(nodes: usize) -> Coordinates<3> {
    (0..nodes)
        .map(|node| [node as f64, 0.5 * node as f64, (node * node % 7) as f64])
        .collect::<Vec<_>>()
        .into()
}

fn write_and_read(mesh: &Mesh<3>, file: &str) -> Mesh<3> {
    mesh.write(Output::Exodus(ExodusFormat::Classic(file)))
        .unwrap();
    Mesh::<3>::try_from(Input::Exodus(file)).unwrap()
}

fn assert_polyhedral(read: &Connectivity, expected: &(Vec<Vec<usize>>, Vec<Vec<usize>>)) {
    match read {
        Connectivity::Polyhedral(poly) => {
            assert!(poly.iter().eq(expected.0.iter()));
            assert_eq!(poly.faces_nodes(), expected.1);
        }
        _ => panic!("expected Polyhedral block"),
    }
}

#[test]
fn round_trip_primitive_and_polyhedral_blocks() {
    let polyhedron = cube(cube_nodes(4));
    let original = Mesh::from((
        Connectivities::from((
            vec![
                Connectivity::Tetrahedral(vec![[0, 1, 2, 3]].into()),
                Connectivity::Polyhedral((polyhedron.0.clone(), polyhedron.1.clone()).into()),
            ],
            vec![3, 7],
        )),
        mixed_coordinates(12),
    ));
    let read = write_and_read(&original, "target/read_exodus_primitive_and_polyhedral.exo");
    assert_eq!(read.connectivities.labels(), Some([3, 7].as_slice()));
    assert_eq!(read.coordinates(), original.coordinates());
    match &read.connectivities()[0] {
        Connectivity::Tetrahedral(tets) => assert!(tets.iter().eq([[0, 1, 2, 3]].iter())),
        _ => panic!("expected Tetrahedral block"),
    }
    assert_polyhedral(&read.connectivities()[1], &polyhedron);
}

#[test]
fn round_trip_polyhedral_blocks_around_a_primitive_block() {
    let (first, second) = (cube(cube_nodes(0)), cube(cube_nodes(12)));
    let original = Mesh::from((
        vec![
            Connectivity::Polyhedral((first.0.clone(), first.1.clone()).into()),
            Connectivity::Tetrahedral(vec![[8, 9, 10, 11]].into()),
            Connectivity::Polyhedral((second.0.clone(), second.1.clone()).into()),
        ],
        mixed_coordinates(20),
    ));
    let read = write_and_read(
        &original,
        "target/read_exodus_polyhedral_around_primitive.exo",
    );
    assert_eq!(read.connectivities().len(), 3);
    assert_polyhedral(&read.connectivities()[0], &first);
    assert_polyhedral(&read.connectivities()[2], &second);
    match &read.connectivities()[1] {
        Connectivity::Tetrahedral(tets) => assert!(tets.iter().eq([[8, 9, 10, 11]].iter())),
        _ => panic!("expected Tetrahedral block"),
    }
}

#[test]
fn face_blocks_are_numbered_among_the_face_blocks() {
    let (first, second) = (cube(cube_nodes(0)), cube(cube_nodes(12)));
    let file = "target/read_exodus_face_block_numbering.exo";
    Mesh::from((
        Connectivities::from((
            vec![
                Connectivity::Polyhedral((first.0.clone(), first.1.clone()).into()),
                Connectivity::Tetrahedral(vec![[8, 9, 10, 11]].into()),
                Connectivity::Polyhedral((second.0.clone(), second.1.clone()).into()),
            ],
            vec![4, 5, 6],
        )),
        mixed_coordinates(20),
    ))
    .write(Output::Exodus(ExodusFormat::Classic(file)))
    .unwrap();
    let netcdf = NetCDF::open(file).unwrap();
    assert_eq!(netcdf.dimension_length("num_el_blk").unwrap(), 3);
    assert_eq!(netcdf.dimension_length("num_fa_blk").unwrap(), 2);
    assert_eq!(
        netcdf.get_variable::<i32>("fa_prop1", 2).unwrap(),
        vec![4, 6]
    );
    assert_eq!(netcdf.dimension_length("num_fa_in_blk1").unwrap(), 6);
    assert_eq!(netcdf.dimension_length("num_fa_in_blk2").unwrap(), 6);
    assert_eq!(netcdf.try_dimension_length("num_fa_in_blk3").unwrap(), None);
    let second_faces = netcdf.get_variable::<i32>("facconn3", 6).unwrap();
    assert_eq!(second_faces.iter().min(), Some(&7));
    assert_eq!(second_faces.iter().max(), Some(&12));
}

#[test]
fn face_numbers_start_after_the_faces_before_them_or_from_one() {
    assert_eq!(face_offset(&[1, 2, 6, 3], 0), 0);
    assert_eq!(face_offset(&[7, 8, 12, 9], 6), 6);
    assert_eq!(face_offset(&[1, 2, 6, 3], 6), 0);
    assert_eq!(face_offset(&[], 6), 0);
}

#[test]
fn netcdf4_matches_classic() {
    let classic_path = format!("{FIXTURES}/sphere_classic.nc");
    let classic = Mesh::<3>::try_from(Input::Exodus(&classic_path)).unwrap();
    for name in ["sphere_nc4.nc", "sphere_nc4_zip.nc"] {
        let path = format!("{FIXTURES}/{name}");
        let read = Mesh::<3>::try_from(Input::Exodus(&path)).unwrap();
        assert_eq!(
            read.coordinates(),
            classic.coordinates(),
            "{name} coordinates"
        );
        match (&read.connectivities()[0], &classic.connectivities()[0]) {
            (Connectivity::Hexahedral(a), Connectivity::Hexahedral(b)) => {
                assert!(a.iter().eq(b.iter()), "{name} connectivity")
            }
            _ => panic!("expected Hexahedral block"),
        }
    }
}

#[test]
fn netcdf4_write_then_read_round_trip() {
    let original = mesh();
    original
        .write(Output::Exodus(ExodusFormat::Netcdf4 {
            path: "target/exodus_netcdf4_round_trip.exo",
            threads: 2,
        }))
        .unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus("target/exodus_netcdf4_round_trip.exo")).unwrap();
    assert_eq!(read.coordinates(), &Coordinates::from(COORDINATES));
    match &read.connectivities()[0] {
        Connectivity::Triangular(triangles) => assert!(triangles.iter().eq(CONNECTIVITY.iter())),
        _ => panic!("expected Triangular block"),
    }
}

fn nul_terminated_exo_bytes(elem_type: &[u8], nodes: usize) -> Vec<u8> {
    const NC_DIMENSION: u32 = 0x0A;
    const NC_VARIABLE: u32 = 0x0B;
    const NC_ATTRIBUTE: u32 = 0x0C;
    const NC_CHAR: i32 = 2;
    const NC_INT: i32 = 4;
    const NC_DOUBLE: i32 = 6;
    let mut b = Vec::<u8>::new();
    let u32be = |b: &mut Vec<u8>, x: u32| b.extend_from_slice(&x.to_be_bytes());
    let name = |b: &mut Vec<u8>, s: &str| {
        b.extend_from_slice(&(s.len() as u32).to_be_bytes());
        b.extend_from_slice(s.as_bytes());
        while !b.len().is_multiple_of(4) {
            b.push(0);
        }
    };
    b.extend_from_slice(b"CDF\x01");
    u32be(&mut b, 0);
    let dims: [(&str, u32); 6] = [
        ("num_dim", 3),
        ("num_nodes", nodes as u32),
        ("num_elem", 1),
        ("num_el_blk", 1),
        ("num_el_in_blk1", 1),
        ("num_nod_per_el1", nodes as u32),
    ];
    u32be(&mut b, NC_DIMENSION);
    u32be(&mut b, dims.len() as u32);
    for &(n, len) in dims.iter() {
        name(&mut b, n);
        u32be(&mut b, len);
    }
    u32be(&mut b, 0);
    u32be(&mut b, 0);
    u32be(&mut b, NC_VARIABLE);
    u32be(&mut b, 5);
    let mut begin_at: Vec<usize> = Vec::new();
    let mut vsizes: Vec<usize> = Vec::new();
    let fixed_var = |b: &mut Vec<u8>,
                     begin_at: &mut Vec<usize>,
                     vsizes: &mut Vec<usize>,
                     n: &str,
                     dimids: &[u32],
                     xtype: i32,
                     vsize: usize| {
        name(b, n);
        b.extend_from_slice(&(dimids.len() as u32).to_be_bytes());
        for &d in dimids {
            b.extend_from_slice(&d.to_be_bytes());
        }
        if n == "connect1" {
            b.extend_from_slice(&NC_ATTRIBUTE.to_be_bytes());
            b.extend_from_slice(&1u32.to_be_bytes());
            name(b, "elem_type");
            b.extend_from_slice(&NC_CHAR.to_be_bytes());
            b.extend_from_slice(&(elem_type.len() as u32 + 1).to_be_bytes());
            b.extend_from_slice(elem_type);
            b.push(0);
            while !b.len().is_multiple_of(4) {
                b.push(0);
            }
        } else {
            b.extend_from_slice(&0u32.to_be_bytes());
            b.extend_from_slice(&0u32.to_be_bytes());
        }
        b.extend_from_slice(&xtype.to_be_bytes());
        b.extend_from_slice(&(vsize as u32).to_be_bytes());
        begin_at.push(b.len());
        b.extend_from_slice(&0u32.to_be_bytes());
        vsizes.push(vsize);
    };
    fixed_var(
        &mut b,
        &mut begin_at,
        &mut vsizes,
        "eb_prop1",
        &[3],
        NC_INT,
        4,
    );
    fixed_var(
        &mut b,
        &mut begin_at,
        &mut vsizes,
        "coordx",
        &[1],
        NC_DOUBLE,
        8 * nodes,
    );
    fixed_var(
        &mut b,
        &mut begin_at,
        &mut vsizes,
        "coordy",
        &[1],
        NC_DOUBLE,
        8 * nodes,
    );
    fixed_var(
        &mut b,
        &mut begin_at,
        &mut vsizes,
        "coordz",
        &[1],
        NC_DOUBLE,
        8 * nodes,
    );
    fixed_var(
        &mut b,
        &mut begin_at,
        &mut vsizes,
        "connect1",
        &[4, 5],
        NC_INT,
        4 * nodes,
    );
    let mut offset = b.len();
    for (&pos, &vsize) in begin_at.iter().zip(&vsizes) {
        b[pos..pos + 4].copy_from_slice(&(offset as u32).to_be_bytes());
        offset += vsize.div_ceil(4) * 4;
    }
    b.extend_from_slice(&1i32.to_be_bytes());
    for axis in 0..3 {
        for node in 0..nodes {
            let v = if node == axis + 1 { 1.0_f64 } else { 0.0 };
            b.extend_from_slice(&v.to_bits().to_be_bytes());
        }
    }
    for node in 1..=nodes as i32 {
        b.extend_from_slice(&node.to_be_bytes());
    }
    b
}

#[test]
fn reads_nul_terminated_tetra() {
    let path = "target/read_exodus_nul_terminated_tetra.exo";
    std::fs::write(path, nul_terminated_exo_bytes(b"TETRA", 4)).unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus(path)).unwrap();
    assert_eq!(
        read.coordinates(),
        &Coordinates::from([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    );
    match &read.connectivities()[0] {
        Connectivity::Tetrahedral(tets) => assert!(tets.iter().eq([[0, 1, 2, 3]].iter())),
        _ => panic!("expected Tetrahedral block"),
    }
}

#[test]
fn reads_bare_pyramid() {
    let path = "target/read_exodus_bare_pyramid.exo";
    std::fs::write(path, nul_terminated_exo_bytes(b"PYRAMID", 5)).unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus(path)).unwrap();
    match &read.connectivities()[0] {
        Connectivity::Pyramidal(pyramids) => {
            assert!(pyramids.iter().eq([[0, 1, 2, 3, 4]].iter()))
        }
        _ => panic!("expected Pyramidal block"),
    }
}

#[test]
fn reads_bare_hex() {
    let path = "target/read_exodus_bare_hex.exo";
    std::fs::write(path, nul_terminated_exo_bytes(b"HEX", 8)).unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus(path)).unwrap();
    match &read.connectivities()[0] {
        Connectivity::Hexahedral(hexes) => {
            assert!(hexes.iter().eq([[0, 1, 2, 3, 4, 5, 6, 7]].iter()))
        }
        _ => panic!("expected Hexahedral block"),
    }
}

#[test]
fn reads_bare_wedge() {
    let path = "target/read_exodus_bare_wedge.exo";
    std::fs::write(path, nul_terminated_exo_bytes(b"WEDGE", 6)).unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus(path)).unwrap();
    match &read.connectivities()[0] {
        Connectivity::Wedge(wedges) => assert!(wedges.iter().eq([[0, 1, 2, 3, 4, 5]].iter())),
        _ => panic!("expected Wedge block"),
    }
}
