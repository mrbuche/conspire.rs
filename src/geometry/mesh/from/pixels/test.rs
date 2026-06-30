use crate::geometry::{grid::Pixels, mesh::Mesh};

fn first_element(mesh: &Mesh<2>, block: usize) -> &[usize] {
    mesh.iter().nth(block).unwrap().iter().next().unwrap()
}

#[test]
fn all_pixels_grouped_by_material() {
    let mesh = Mesh::from_pixels(Pixels::new(vec![1u8, 2], [2, 1]), None);
    assert_eq!(mesh.number_of_elements(), 2);
    assert_eq!(mesh.number_of_nodes(), 6);
    assert_eq!(mesh.number_of_element_blocks(), 2);
    assert_eq!(first_element(&mesh, 0), [0, 1, 4, 3]);
    assert_eq!(first_element(&mesh, 1), [1, 2, 5, 4]);
}

#[test]
fn removes_material() {
    let mesh = Mesh::from_pixels(Pixels::new(vec![1u8, 2], [2, 1]), Some(&[2]));
    assert_eq!(mesh.number_of_elements(), 1);
    assert_eq!(mesh.number_of_nodes(), 4);
    assert_eq!(mesh.number_of_element_blocks(), 1);
    assert_eq!(first_element(&mesh, 0), [0, 1, 3, 2]);
}
