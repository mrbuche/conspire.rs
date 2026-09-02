use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Connectivities, Connectivity, ExodusFormat, Input, Mesh, NodeSets, Output, SideSets,
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
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_block_numbers.exo",
        )))
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
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_element_numbers.exo",
        )))
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
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_node_numbers.exo",
        )))
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
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_node_sets.exo",
        )))
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
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_node_set_numbers.exo",
        )))
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
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_side_sets.exo",
        )))
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
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_side_set_numbers.exo",
        )))
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
        .write(Output::Exodus(ExodusFormat::Classic(
            "target/read_exodus_side_sets_custom_elements.exo",
        )))
        .unwrap();
    let read = Mesh::<3>::try_from(Input::Exodus(
        "target/read_exodus_side_sets_custom_elements.exo",
    ))
    .unwrap();
    assert_eq!(read.side_sets(), &[vec![(0, 0), (1, 2)]]);
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

fn nul_terminated_tetra_exo_bytes() -> Vec<u8> {
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
        while b.len() % 4 != 0 {
            b.push(0);
        }
    };
    b.extend_from_slice(b"CDF\x01");
    u32be(&mut b, 0);
    let dims: [(&str, u32); 6] = [
        ("num_dim", 3),
        ("num_nodes", 4),
        ("num_elem", 1),
        ("num_el_blk", 1),
        ("num_el_in_blk1", 1),
        ("num_nod_per_el1", 4),
    ];
    u32be(&mut b, NC_DIMENSION);
    u32be(&mut b, dims.len() as u32);
    for (n, len) in dims {
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
            b.extend_from_slice(&6u32.to_be_bytes());
            b.extend_from_slice(b"TETRA\x00");
            while b.len() % 4 != 0 {
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
        32,
    );
    fixed_var(
        &mut b,
        &mut begin_at,
        &mut vsizes,
        "coordy",
        &[1],
        NC_DOUBLE,
        32,
    );
    fixed_var(
        &mut b,
        &mut begin_at,
        &mut vsizes,
        "coordz",
        &[1],
        NC_DOUBLE,
        32,
    );
    fixed_var(
        &mut b,
        &mut begin_at,
        &mut vsizes,
        "connect1",
        &[4, 5],
        NC_INT,
        16,
    );
    let mut offset = b.len();
    for (&pos, &vsize) in begin_at.iter().zip(&vsizes) {
        b[pos..pos + 4].copy_from_slice(&(offset as u32).to_be_bytes());
        offset += vsize.div_ceil(4) * 4;
    }
    b.extend_from_slice(&1i32.to_be_bytes());
    for coord in [
        [0.0_f64, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ] {
        for v in coord {
            b.extend_from_slice(&v.to_bits().to_be_bytes());
        }
    }
    for node in [1i32, 2, 3, 4] {
        b.extend_from_slice(&node.to_be_bytes());
    }
    b
}

#[test]
fn reads_nul_terminated_tetra() {
    let path = "target/read_exodus_nul_terminated_tetra.exo";
    std::fs::write(path, nul_terminated_tetra_exo_bytes()).unwrap();
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
