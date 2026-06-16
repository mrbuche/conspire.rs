use crate::geometry::{grid::Voxels, mesh::Mesh};

fn first_element(mesh: &Mesh<3>, block: usize) -> &[usize] {
    mesh.iter().nth(block).unwrap().iter().next().unwrap()
}

#[test]
fn all_voxels_grouped_by_material() {
    let mesh = Mesh::from_voxels(Voxels::new(vec![1u8, 2], [2, 1, 1]), None);
    assert_eq!(mesh.number_of_elements(), 2);
    assert_eq!(mesh.number_of_nodes(), 12);
    assert_eq!(mesh.number_of_element_blocks(), 2);
    assert_eq!(first_element(&mesh, 0), [0, 1, 4, 3, 6, 7, 10, 9]);
    assert_eq!(first_element(&mesh, 1), [1, 2, 5, 4, 7, 8, 11, 10]);
}

#[test]
fn removes_material() {
    let mesh = Mesh::from_voxels(Voxels::new(vec![1u8, 2], [2, 1, 1]), Some(&[2]));
    assert_eq!(mesh.number_of_elements(), 1);
    assert_eq!(mesh.number_of_nodes(), 8);
    assert_eq!(mesh.number_of_element_blocks(), 1);
    assert_eq!(first_element(&mesh, 0), [0, 1, 3, 2, 4, 5, 7, 6]);
}
