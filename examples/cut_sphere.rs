use conspire::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh, Output, Tessellation, Vtk},
        ntree::Balancing,
    },
    io::{Write, write::Compression},
};
use std::collections::HashMap;

fn midpoint(
    a: usize,
    b: usize,
    coordinates: &mut Vec<[f64; 3]>,
    cache: &mut HashMap<[usize; 2], usize>,
) -> usize {
    let key = if a < b { [a, b] } else { [b, a] };
    *cache.entry(key).or_insert_with(|| {
        let (p, q) = (coordinates[a], coordinates[b]);
        let m = [p[0] + q[0], p[1] + q[1], p[2] + q[2]];
        let norm = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
        coordinates.push([m[0] / norm, m[1] / norm, m[2] / norm]);
        coordinates.len() - 1
    })
}

fn sphere(refinements: usize) -> Tessellation {
    let mut coordinates = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let mut faces = vec![
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];
    (0..refinements).for_each(|_| {
        let mut cache = HashMap::new();
        faces = faces
            .iter()
            .flat_map(|&[a, b, c]| {
                let ab = midpoint(a, b, &mut coordinates, &mut cache);
                let bc = midpoint(b, c, &mut coordinates, &mut cache);
                let ca = midpoint(c, a, &mut coordinates, &mut cache);
                [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
            })
            .collect()
    });
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(coordinates),
    )))
}

fn main() -> Result<(), std::io::Error> {
    let tessellation = sphere(4);
    tessellation.write("sphere.stl")?;
    let mesh = tessellation.cut(Balancing::Strong, 16.0).unwrap();
    mesh.iter().try_for_each(|block| match block {
        Connectivity::Hexahedral(_) => {
            let hexes: Vec<[usize; 8]> = block
                .iter()
                .map(|element| std::array::from_fn(|i| element[i]))
                .collect();
            Mesh::from((
                vec![Connectivity::Hexahedral(hexes.into())],
                mesh.coordinates().clone(),
            ))
            .write(Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(
                "hexes.vtu",
            ))))
        }
        Connectivity::Polyhedral(polyhedra) => {
            let triangles: Vec<[usize; 3]> = polyhedra
                .faces_nodes()
                .iter()
                .flat_map(|face| {
                    (1..face.len() - 1)
                        .map(|i| [face[0], face[i], face[i + 1]])
                        .collect::<Vec<_>>()
                })
                .collect();
            Tessellation::from(Mesh::from((
                vec![Connectivity::Triangular(triangles.into())],
                mesh.coordinates().clone(),
            )))
            .write("poly_faces.stl")?;
            let polys = Mesh::from((
                vec![Connectivity::Polyhedral(
                    (
                        polyhedra.elements_faces().to_vec(),
                        polyhedra.faces_nodes().to_vec(),
                    )
                        .into(),
                )],
                mesh.coordinates().clone(),
            ));
            #[cfg(feature = "netcdf")]
            polys.write(Output::Exodus("polys.exo"))?;
            polys.write(Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(
                "polys.vtu",
            ))))
        }
        _ => Ok(()),
    })?;
    println!("wrote sphere.stl, hexes.vtu, poly_faces.stl, polys.vtu, polys.exo");
    Ok(())
}
