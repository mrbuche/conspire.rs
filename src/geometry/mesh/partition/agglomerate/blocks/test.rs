use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivities, Connectivity, Mesh, Partition},
    },
    math::Set,
};

fn star() -> Vec<[usize; 4]> {
    vec![[0, 1, 2, 4], [0, 1, 4, 3], [0, 4, 2, 3], [4, 1, 2, 3]]
}

fn star_coordinates() -> Coordinates<3> {
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.2, 0.2, 0.2],
    ]
    .into()
}

fn mesh(blocks: Vec<Connectivity>, numbers: Option<Vec<usize>>) -> Mesh<3> {
    let connectivities = match numbers {
        Some(numbers) => Connectivities::from((blocks, numbers)),
        None => Connectivities::from(blocks),
    };
    (connectivities, Set::from(star_coordinates())).into()
}

fn tets(elements: Vec<[usize; 4]>) -> Connectivity {
    Connectivity::Tetrahedral(elements.into())
}

fn blocks(mesh: &Mesh<3>) -> Vec<(&'static str, usize)> {
    mesh.connectivities()
        .iter()
        .map(|block| {
            let kind = match block {
                Connectivity::Tetrahedral(_) => "tetrahedra",
                Connectivity::Hexahedral(_) => "hexahedra",
                Connectivity::Polyhedral(_) => "polyhedra",
                _ => "other",
            };
            (kind, block.number_of_elements())
        })
        .collect()
}

#[test]
fn merging_every_element_leaves_one_polyhedron_and_drops_the_interior_node() {
    let mesh = mesh(vec![tets(star())], None);
    let result = Partition::new(&mesh, vec![0, 0, 0, 0])
        .agglomerate_blocks(&mesh)
        .unwrap();
    assert_eq!(blocks(&result), [("polyhedra", 1)]);
    assert_eq!(result.number_of_nodes(), 4);
}

#[test]
fn elements_alone_in_their_part_stay_in_a_block_of_their_own_type() {
    let mesh = mesh(vec![tets(star())], None);
    let result = Partition::new(&mesh, vec![0, 0, 1, 2])
        .agglomerate_blocks(&mesh)
        .unwrap();
    assert_eq!(blocks(&result), [("tetrahedra", 2), ("polyhedra", 1)]);
    assert_eq!(result.number_of_nodes(), 5);
}

#[test]
fn nothing_merged_gives_back_the_same_blocks() {
    let mesh = mesh(vec![tets(star())], None);
    let result = Partition::new(&mesh, vec![0, 1, 2, 3])
        .agglomerate_blocks(&mesh)
        .unwrap();
    assert_eq!(blocks(&result), [("tetrahedra", 4)]);
    assert_eq!(result.number_of_nodes(), 5);
    assert!(result.blocks().is_none());
}

#[test]
fn each_block_gets_its_own_polyhedral_block_and_the_numbers_follow() {
    let mesh = mesh(
        vec![
            tets(vec![star()[0], star()[1]]),
            tets(vec![star()[2], star()[3]]),
        ],
        Some(vec![5, 7]),
    );
    let result = Partition::new(&mesh, vec![0, 0, 1, 2])
        .agglomerate_blocks(&mesh)
        .unwrap();
    assert_eq!(blocks(&result), [("polyhedra", 1), ("tetrahedra", 2)]);
    assert_eq!(result.blocks(), Some(&[8, 7][..]));
}

#[test]
fn every_block_with_merges_takes_the_next_number_in_turn() {
    let mesh = mesh(
        vec![
            tets(vec![star()[0], star()[1]]),
            tets(vec![star()[2], star()[3]]),
        ],
        Some(vec![5, 7]),
    );
    let result = Partition::new(&mesh, vec![0, 0, 1, 1])
        .agglomerate_blocks(&mesh)
        .unwrap();
    assert_eq!(blocks(&result), [("polyhedra", 1), ("polyhedra", 1)]);
    assert_eq!(result.blocks(), Some(&[8, 9][..]));
}

#[test]
fn a_block_with_some_merged_elements_keeps_its_number_and_the_polyhedra_get_a_new_one() {
    let mesh = mesh(vec![tets(star())], Some(vec![5]));
    let result = Partition::new(&mesh, vec![0, 0, 1, 2])
        .agglomerate_blocks(&mesh)
        .unwrap();
    assert_eq!(blocks(&result), [("tetrahedra", 2), ("polyhedra", 1)]);
    assert_eq!(result.blocks(), Some(&[5, 6][..]));
}

#[test]
fn parts_in_several_blocks_are_an_error() {
    let mesh = mesh(
        vec![
            tets(vec![star()[0], star()[1]]),
            tets(vec![star()[2], star()[3]]),
        ],
        None,
    );
    let result = Partition::new(&mesh, vec![0, 1, 0, 2]).agglomerate_blocks(&mesh);
    assert_eq!(result.err(), Some("a part has elements in several blocks"));
}

#[test]
fn hexahedra_alone_stay_hexahedra_and_a_pair_becomes_a_polyhedron() {
    let hexes = Connectivity::Hexahedral(
        vec![[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]].into(),
    );
    let coordinates = Coordinates::from(
        (0..12)
            .map(|node| {
                [
                    (node % 3) as f64,
                    ((node / 3) % 2) as f64,
                    (node / 6) as f64,
                ]
            })
            .collect::<Vec<_>>(),
    );
    let mesh: Mesh<3> = (Connectivities::from(vec![hexes]), Set::from(coordinates)).into();
    let alone = Partition::new(&mesh, vec![0, 1])
        .agglomerate_blocks(&mesh)
        .unwrap();
    assert_eq!(blocks(&alone), [("hexahedra", 2)]);
    let joined = Partition::new(&mesh, vec![0, 0])
        .agglomerate_blocks(&mesh)
        .unwrap();
    assert_eq!(blocks(&joined), [("polyhedra", 1)]);
    assert_eq!(joined.number_of_nodes(), 12);
}

#[test]
fn a_polyhedral_block_stays_one_polyhedral_block() {
    let source = mesh(vec![tets(star())], None);
    let polyhedra = Partition::new(&source, vec![0, 0, 1, 2])
        .agglomerate(&source)
        .unwrap();
    assert_eq!(blocks(&polyhedra), [("polyhedra", 3)]);
    let result = Partition::new(&polyhedra, vec![0, 0, 1])
        .agglomerate_blocks(&polyhedra)
        .unwrap();
    assert_eq!(blocks(&result), [("polyhedra", 2)]);
    let apart = Partition::new(&polyhedra, vec![0, 1, 2])
        .agglomerate_blocks(&polyhedra)
        .unwrap();
    assert_eq!(blocks(&apart), [("polyhedra", 3)]);
}
