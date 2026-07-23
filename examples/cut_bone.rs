use conspire::geometry::{
    Coordinates,
    mesh::{Connectivity, Mesh, Output, Tessellation, Vtk},
    ntree::Balancing,
};
use conspire::io::{Write, write::Compression};
use conspire::math::Tensor;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
    time::Instant,
};

fn read_stl(path: &str) -> Tessellation {
    let mut reader = BufReader::new(File::open(path).unwrap());
    let mut header = [0_u8; 84];
    reader.read_exact(&mut header).unwrap();
    let count = u32::from_le_bytes(header[80..84].try_into().unwrap());
    let mut coordinates = vec![];
    let mut lookup = HashMap::new();
    let mut faces = Vec::with_capacity(count as usize);
    let mut buffer = [0_u8; 50];
    for _ in 0..count {
        reader.read_exact(&mut buffer).unwrap();
        let node = |k: usize| -> [f32; 3] {
            std::array::from_fn(|i| {
                f32::from_le_bytes(
                    buffer[12 + 12 * k + 4 * i..16 + 12 * k + 4 * i]
                        .try_into()
                        .unwrap(),
                )
            })
        };
        let face: [usize; 3] = std::array::from_fn(|k| {
            let point = node(k);
            let key = point.map(f32::to_bits);
            *lookup.entry(key).or_insert_with(|| {
                coordinates.push([point[0] as f64, point[1] as f64, point[2] as f64]);
                coordinates.len() - 1
            })
        });
        faces.push(face);
    }
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(coordinates),
    )))
}

fn main() {
    let tessellation = read_stl("bone_tri.stl");
    println!(
        "surface: {} triangles, {} nodes\n",
        tessellation.mesh().number_of_elements(),
        tessellation.mesh().coordinates().len()
    );
    for scale in [64.0, 32.0, 16.0, 8.0, 4.0, 3.0, 2.0] {
        println!("scale = {scale}");
        let t = Instant::now();
        match tessellation.cut(Balancing::Strong, scale) {
            Ok(mesh) => {
                println!(
                    "  -> {} elements, {} nodes, wall {:?}",
                    mesh.number_of_elements(),
                    mesh.coordinates().len(),
                    t.elapsed()
                );
                let path = format!("target/cut_bone/scale_{scale}.vtu");
                mesh.write(Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(&path))))
                    .unwrap();
                println!("  -> wrote {path}\n");
            }
            Err(e) => println!("  -> FAILED: {e}, wall {:?}\n", t.elapsed()),
        }
    }
}
