use crate::geometry::{grid::Voxels, mesh::Tessellation};

#[test]
fn single_voxel_is_a_cube() {
    let tessellation = Tessellation::from(Voxels::new(vec![1], [1, 1, 1]));
    assert_eq!(tessellation.mesh().number_of_nodes(), 8);
    assert_eq!(tessellation.mesh().number_of_elements(), 12);
}

#[test]
fn void_neighbor_is_faceted_but_void_is_not() {
    let tessellation = Tessellation::from(Voxels::new(vec![1, 0], [2, 1, 1]));
    assert_eq!(tessellation.mesh().number_of_nodes(), 8);
    assert_eq!(tessellation.mesh().number_of_elements(), 12);
}

#[test]
fn shared_interface_is_suppressed_and_vertices_welded() {
    let tessellation = Tessellation::from(Voxels::new(vec![1, 1], [2, 1, 1]));
    assert_eq!(tessellation.mesh().number_of_nodes(), 12);
    assert_eq!(tessellation.mesh().number_of_elements(), 20);
}

#[test]
fn material_interface_is_faceted_once() {
    let tessellation = Tessellation::from(Voxels::new(vec![1, 2], [2, 1, 1]));
    assert_eq!(tessellation.mesh().number_of_nodes(), 12);
    assert_eq!(tessellation.mesh().number_of_elements(), 22);
}
