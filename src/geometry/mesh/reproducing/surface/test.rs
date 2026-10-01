use crate::{
    geometry::mesh::{Basis, Mesh, test::tetrahedra},
    math::Quantity,
    units::Length,
};

const TOLERANCE: f64 = 1e-9;

fn basis(mesh: &Mesh<3>) -> Basis {
    let seeds = mesh.sample(Quantity::<Length>::new(0.4), 2);
    mesh.reproducing_basis(&seeds, Quantity::new(1.1), 1, 1)
        .unwrap()
}

fn x(mesh: &Mesh<3>, node: usize) -> f64 {
    mesh.coordinates()[node][0].value()
}

#[test]
fn integrals_over_the_whole_surface_add_up_to_its_area() {
    let mesh = tetrahedra(6);
    let integrals = mesh
        .face_integrals(&basis(&mesh), &mesh.exterior_faces())
        .unwrap();
    let total: f64 = integrals.iter().map(|i| i.value()).sum();
    assert!((total - 6.0).abs() < TOLERANCE, "{total}");
}

#[test]
fn normals_over_a_closed_surface_cancel() {
    let mesh = tetrahedra(6);
    let integrals = mesh
        .face_normal_integrals(&basis(&mesh), &mesh.exterior_faces())
        .unwrap();
    for k in 0..3 {
        let total: f64 = integrals.iter().map(|i| i[k].value()).sum();
        assert!(total.abs() < TOLERANCE, "axis {k}: {total}");
    }
}

#[test]
fn surface_integral_of_the_normal_is_the_volume_integral_of_the_gradient() {
    let mesh = tetrahedra(6);
    let basis = basis(&mesh);
    let surface = mesh
        .face_normal_integrals(&basis, &mesh.exterior_faces())
        .unwrap();
    let elements: Vec<usize> = (0..mesh.number_of_elements()).collect();
    let simplices = mesh.simplices_over::<4>(&elements).unwrap();
    for (function, values) in basis.values.iter().enumerate() {
        let mut volume = [0.0; 3];
        for simplex in &simplices {
            let mut gradient = [0.0; 3];
            for &(node, value) in values {
                if let Some(a) = simplex.nodes.iter().position(|&n| n == node) {
                    (0..3).for_each(|k| gradient[k] += value * simplex.gradients[a][k]);
                }
            }
            (0..3).for_each(|k| volume[k] += simplex.volume * gradient[k]);
        }
        for k in 0..3 {
            assert!(
                (surface[function][k].value() - volume[k]).abs() < TOLERANCE,
                "function {function}, axis {k}: {} vs {}",
                surface[function][k].value(),
                volume[k]
            );
        }
    }
}

#[test]
fn one_face_of_the_cube() {
    let mesh = tetrahedra(6);
    let basis = basis(&mesh);
    let faces: Vec<Vec<usize>> = mesh
        .exterior_faces()
        .into_iter()
        .filter(|face| face.iter().all(|&node| x(&mesh, node) < 1e-12))
        .collect();
    assert_eq!(faces.len(), 2 * 6 * 6);
    let area: f64 = mesh
        .face_integrals(&basis, &faces)
        .unwrap()
        .iter()
        .map(|i| i.value())
        .sum();
    assert!((area - 1.0).abs() < TOLERANCE, "{area}");
    let normal = mesh.face_normal_integrals(&basis, &faces).unwrap();
    let total: Vec<f64> = (0..3)
        .map(|k| normal.iter().map(|i| i[k].value()).sum())
        .collect();
    assert!((total[0] + 1.0).abs() < TOLERANCE, "{total:?}");
    assert!(total[1].abs() < TOLERANCE && total[2].abs() < TOLERANCE);
}

#[test]
fn only_triangular_faces() {
    let mesh = tetrahedra(6);
    let basis = basis(&mesh);
    let error = "surface integrals require triangular faces";
    assert_eq!(
        mesh.face_integrals(&basis, &[vec![0, 1]]).unwrap_err(),
        error
    );
    assert_eq!(
        mesh.face_normal_integrals(&basis, &[vec![0, 1, 2, 3]])
            .unwrap_err(),
        error
    );
}

#[test]
fn normals_need_faces_on_the_boundary() {
    let mesh = tetrahedra(6);
    let basis = basis(&mesh);
    let exterior: Vec<Vec<usize>> = mesh
        .exterior_faces()
        .into_iter()
        .map(|mut face| {
            face.sort_unstable();
            face
        })
        .collect();
    let element = mesh.iter().flat_map(|block| block.iter()).next().unwrap();
    let interior: Vec<usize> = (0..4)
        .map(|skip| {
            element
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != skip)
                .map(|(_, &node)| node)
                .collect::<Vec<usize>>()
        })
        .map(|mut face| {
            face.sort_unstable();
            face
        })
        .find(|face| !exterior.contains(face))
        .unwrap();
    let faces = [interior];
    assert_eq!(
        mesh.face_normal_integrals(&basis, &faces).unwrap_err(),
        "surface normals require faces on the boundary of the mesh"
    );
    assert!(mesh.face_integrals(&basis, &faces).is_ok());
}
