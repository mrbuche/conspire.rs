use super::FromConnectivities;
use crate::{
    domain::{
        Model, NodalReferenceCoordinates,
        boundary::{BoundaryConditions, Displacement, Solid},
    },
    geometry::mesh::{Connectivity, Mesh, NodeSet, SideSet},
    math::optimize::EqualityConstraint,
};

#[derive(Debug)]
struct NoBlocks;

impl FromConnectivities<3, ()> for NoBlocks {
    fn from_connectivities(
        _: Vec<Connectivity>,
        _: (),
        _: &NodalReferenceCoordinates<3>,
    ) -> Result<Self, String> {
        Ok(Self)
    }
}

fn mesh() -> Mesh<3> {
    (
        vec![Connectivity::Triangular(vec![[0, 1, 2]].into())],
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]].into(),
    )
        .into()
}

#[test]
fn a_model_keeps_the_sets_of_its_mesh() {
    let mut mesh = mesh();
    mesh.set_node_sets(vec![NodeSet::new(vec![0, 1]).with_name("bottom")].into());
    mesh.set_side_sets(vec![SideSet::new(vec![(0, 1)]).with_label(4)].into());
    let model = Model::<NoBlocks, 3>::try_from((mesh, ())).unwrap();
    assert_eq!(model.node_sets().by_name("bottom").unwrap().nodes(), [0, 1]);
    assert_eq!(model.side_sets().by_label(4).unwrap().sides(), [(0, 1)]);
}

#[test]
fn a_model_finds_the_sets_of_its_boundary_conditions() {
    let mut mesh = mesh();
    mesh.set_node_sets(vec![NodeSet::new(vec![0, 2]).with_name("bottom")].into());
    let model = Model::<NoBlocks, 3>::try_from((mesh, ())).unwrap();
    let bottom = model.node_sets().by_name("bottom").unwrap();
    let fixed = BoundaryConditions::<Solid<3>>::new().fix(Displacement::<1>, bottom);
    match fixed.constraint() {
        EqualityConstraint::Fixed(indices) => assert_eq!(indices, [1, 7]),
        _ => panic!("expected a fixed constraint"),
    }
    assert!(model.node_sets().by_name("top").is_none());
}

#[test]
fn a_model_of_a_mesh_without_sets_has_none() {
    let model = Model::<NoBlocks, 3>::try_from((mesh(), ())).unwrap();
    assert!(model.node_sets().is_empty());
    assert!(model.side_sets().is_empty());
}

#[test]
fn a_model_made_from_blocks_and_coordinates_has_no_sets() {
    let coordinates: NodalReferenceCoordinates<3> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]].into();
    let model = Model::from((NoBlocks, coordinates));
    assert!(model.node_sets().is_empty());
    assert!(model.side_sets().is_empty());
}
