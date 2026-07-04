use crate::{
    geometry::{
        grid::{Input, Voxels},
        mesh::Mesh,
    },
    io::{Npy, Write},
};

fn first_element(mesh: &Mesh<3>, block: usize) -> &[usize] {
    mesh.iter().nth(block).unwrap().iter().next().unwrap()
}

fn min_corner(mesh: &Mesh<3>, element: &[usize]) -> [usize; 3] {
    let coordinates = mesh.coordinates();
    let mut min = [usize::MAX; 3];
    element.iter().for_each(|&node| {
        let point = &coordinates[node];
        (0..3).for_each(|axis| min[axis] = min[axis].min(point[axis] as usize))
    });
    min
}

#[test]
fn c_order_npy_meshes_with_correct_orientation() {
    let nel = [2usize, 1, 3];
    let [nx, ny, nz] = nel;
    let path = "target/orient_mesh.npy";
    Npy {
        data: (1..=(nx * ny * nz) as u8).collect(),
        shape: vec![nx, ny, nz],
        fortran_order: false,
    }
    .write(path)
    .unwrap();
    let voxels = Voxels::<u8>::try_from(Input::Npy(path)).unwrap();
    let mesh = Mesh::from_voxels(voxels, None);
    assert_eq!(min_corner(&mesh, first_element(&mesh, 3)), [1, 0, 0]);
    assert_eq!(min_corner(&mesh, first_element(&mesh, 2)), [0, 0, 2]);
    assert_eq!(min_corner(&mesh, first_element(&mesh, 0)), [0, 0, 0]);
    assert_eq!(min_corner(&mesh, first_element(&mesh, 5)), [1, 0, 2]);
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
