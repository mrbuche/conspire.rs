use crate::math::assert::Assert;
use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Connectivity, Mesh,
            test::{CONNECTIVITY, COORDINATES, mesh},
        },
    },
    math::assert::AssertionError,
};

#[test]
fn bounding_boxes_and_centroids() {
    let mesh: Mesh<3> = mesh();
    mesh.bounding_boxes_and_centroids()
        .zip(mesh.bounding_boxes())
        .zip(mesh.centroids())
        .for_each(|(((bounding_box, centroid), bbox), cntrd)| {
            assert_eq!(bounding_box, bbox);
            assert_eq!(centroid, cntrd)
        })
}

#[test]
fn connectivities() {
    let mesh = mesh();
    match &mesh.connectivities()[0] {
        Connectivity::Triangular(triangles) => {
            assert!(triangles.iter().eq(CONNECTIVITY.iter()))
        }
        _ => panic!("expected Triangular block"),
    }
}

#[test]
fn coordinates() -> Result<(), AssertionError> {
    let mesh = mesh();
    let coordinates = Coordinates::from(COORDINATES);
    Assert::eq(mesh.coordinates(), &coordinates)
}

fn blocks() -> Mesh<3> {
    (
        vec![
            Connectivity::Tetrahedral(vec![[0, 1, 2, 3]].into()),
            Connectivity::Tetrahedral(vec![[0, 2, 1, 4], [1, 2, 3, 4]].into()),
            Connectivity::Hexahedral(vec![[0, 1, 2, 3, 4, 5, 6, 7]].into()),
        ],
        vec![[0.0, 0.0, 0.0]; 8].into(),
    )
        .into()
}

#[test]
fn elements_blocks() {
    let mesh = blocks();
    assert_eq!(mesh.elements_blocks(), [0, 1, 1, 2]);
    assert_eq!(mesh.elements_blocks().len(), mesh.number_of_elements());
}

#[test]
fn elements_blocks_of_one_block() {
    let mesh: Mesh<3> = Mesh::from((
        vec![Connectivity::Polyhedral(
            (
                vec![vec![0, 1], vec![1, 2]],
                vec![vec![0, 1, 2], vec![1, 2, 3], vec![2, 3, 4]],
            )
                .into(),
        )],
        vec![[0.0, 0.0, 0.0]; 5].into(),
    ));
    assert_eq!(mesh.elements_blocks(), [0, 0]);
}

#[test]
fn elements_blocks_of_no_blocks() {
    let mesh: Mesh<3> = Mesh::from((Vec::<Connectivity>::new(), Vec::<[f64; 3]>::new().into()));
    assert!(mesh.elements_blocks().is_empty());
}

#[test]
fn number_of_nodes() {
    let mesh = mesh();
    assert_eq!(mesh.number_of_nodes(), COORDINATES.len())
}

#[test]
fn select_nodes() {
    let mesh = Mesh::from((
        vec![Connectivity::Triangular(vec![[0, 1, 2]].into())],
        Coordinates::<3>::from([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    ));
    assert_eq!(mesh.select_nodes(|x| x[0].value() > 0.5).nodes(), [1]);
    assert!(mesh.select_nodes(|x| x[2].value() > 0.5).nodes().is_empty());
}

mod nodes_from {
    use crate::geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh, SideSet},
    };

    fn line(number_of_nodes: usize) -> Coordinates<3> {
        (0..number_of_nodes)
            .map(|node| [node as f64, 0.0, 0.0])
            .collect::<Vec<_>>()
            .into()
    }
    fn two_hexahedra() -> Mesh<3> {
        (
            vec![Connectivity::Hexahedral(
                vec![[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11]].into(),
            )],
            line(12),
        )
            .into()
    }
    #[test]
    fn one_face() {
        let nodes = two_hexahedra().nodes_from(&SideSet::new(vec![(0, 5)]));
        assert_eq!(nodes.nodes(), [4, 5, 6, 7]);
    }
    #[test]
    fn faces_sharing_nodes_are_deduplicated() {
        let nodes = two_hexahedra().nodes_from(&SideSet::new(vec![(0, 5), (1, 4)]));
        assert_eq!(nodes.nodes(), [4, 5, 6, 7]);
    }
    #[test]
    fn several_faces_of_one_element() {
        let nodes = two_hexahedra().nodes_from(&SideSet::new(vec![(1, 5), (1, 4)]));
        assert_eq!(nodes.nodes(), [4, 5, 6, 7, 8, 9, 10, 11]);
    }
    #[test]
    fn polytopal_face() {
        let faces_nodes = vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0]];
        let mesh = Mesh::from((
            vec![Connectivity::Polygonal(
                (vec![vec![0, 1, 2, 3]], faces_nodes).into(),
            )],
            line(4),
        ));
        assert_eq!(mesh.nodes_from(&SideSet::new(vec![(0, 1)])).nodes(), [1, 2]);
    }
    #[test]
    fn keeps_label_and_name() {
        let side_set = SideSet::new(vec![(0, 5)]).with_label(3).with_name("top");
        let nodes = two_hexahedra().nodes_from(&side_set);
        assert_eq!(nodes.label(), Some(3));
        assert_eq!(nodes.name(), Some("top"));
    }
    #[test]
    #[should_panic(expected = "does not have")]
    fn element_the_mesh_does_not_have() {
        two_hexahedra().nodes_from(&SideSet::new(vec![(2, 0)]));
    }
}
