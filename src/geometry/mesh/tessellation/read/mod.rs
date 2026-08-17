#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates, Direction, Directions,
        mesh::{
            Connectivity,
            tessellation::{D, Tessellation},
        },
    },
    io::invalid,
    math::{Tensor, TensorVec},
};
use std::{
    cell::OnceCell,
    collections::HashMap,
    fs::File,
    io::{BufReader, Error as ErrorIO, Read, Seek, SeekFrom},
    path::Path,
    str::SplitWhitespace,
};

impl TryFrom<&Path> for Tessellation {
    type Error = ErrorIO;
    fn try_from(path: &Path) -> Result<Self, Self::Error> {
        let mut file = File::open(path)?;
        if let Some(triangle_count) = binary_triangle_count(&mut file)? {
            read_binary(BufReader::new(file), triangle_count)
        } else {
            file.seek(SeekFrom::Start(0))?;
            let mut contents = String::new();
            BufReader::new(file).read_to_string(&mut contents)?;
            read_ascii(&contents)
        }
    }
}

fn binary_triangle_count(file: &mut File) -> Result<Option<usize>, ErrorIO> {
    let length = file.metadata()?.len();
    if length < 84 {
        return Ok(None);
    }
    let mut header = [0u8; 84];
    file.read_exact(&mut header)?;
    let triangle_count = u32::from_le_bytes([header[80], header[81], header[82], header[83]]);
    Ok((84 + 50 * triangle_count as u64 == length).then_some(triangle_count as usize))
}

fn read_binary<R: Read>(mut reader: R, triangle_count: usize) -> Result<Tessellation, ErrorIO> {
    let mut builder = Builder::with_capacity(triangle_count);
    (0..triangle_count).try_for_each(|_| {
        let normal = read_vec3_f32(&mut reader)?;
        let vertices = [
            read_vec3_f32(&mut reader)?,
            read_vec3_f32(&mut reader)?,
            read_vec3_f32(&mut reader)?,
        ];
        let mut attribute_byte_count = [0u8; 2];
        reader.read_exact(&mut attribute_byte_count)?;
        builder.push(normal, vertices);
        Ok::<(), ErrorIO>(())
    })?;
    Ok(builder.into())
}

fn read_ascii(contents: &str) -> Result<Tessellation, ErrorIO> {
    let mut builder = Builder::with_capacity(contents.len() / 200);
    let mut tokens = contents.split_whitespace();
    while let Some(token) = tokens.next() {
        if token == "facet" {
            expect(&mut tokens, "normal")?;
            let normal = read_vec3_ascii(&mut tokens)?;
            expect(&mut tokens, "outer")?;
            expect(&mut tokens, "loop")?;
            let mut vertices = [[0.0; D]; 3];
            vertices.iter_mut().try_for_each(|vertex| {
                expect(&mut tokens, "vertex")?;
                *vertex = read_vec3_ascii(&mut tokens)?;
                Ok::<(), ErrorIO>(())
            })?;
            expect(&mut tokens, "endloop")?;
            expect(&mut tokens, "endfacet")?;
            builder.push(normal, vertices);
        }
    }
    Ok(builder.into())
}

fn expect(tokens: &mut SplitWhitespace, keyword: &str) -> Result<(), ErrorIO> {
    match tokens.next() {
        Some(token) if token.eq_ignore_ascii_case(keyword) => Ok(()),
        token => Err(invalid(format!(
            "Expected {keyword} but found {}",
            token.unwrap_or("end of file")
        ))),
    }
}

fn read_vec3_ascii(tokens: &mut SplitWhitespace) -> Result<[f64; D], ErrorIO> {
    let mut vector = [0.0; D];
    vector.iter_mut().try_for_each(|entry| {
        match tokens.next() {
            Some(token) => {
                *entry = token
                    .parse()
                    .map_err(|_| invalid(format!("Expected a number but found {token}")))?
            }
            None => {
                return Err(invalid(
                    "Expected a number but found end of file".to_string(),
                ));
            }
        }
        Ok(())
    })?;
    Ok(vector)
}

fn read_vec3_f32<R: Read>(reader: &mut R) -> Result<[f64; D], ErrorIO> {
    Ok([read_f32(reader)?, read_f32(reader)?, read_f32(reader)?])
}

fn read_f32<R: Read>(reader: &mut R) -> Result<f64, ErrorIO> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes) as f64)
}

struct Builder {
    connectivity: Vec<[usize; D]>,
    normals: Directions<D>,
    vertex_map: HashMap<[u64; D], usize>,
    vertices: Coordinates<D>,
}

impl Builder {
    fn with_capacity(triangle_count: usize) -> Self {
        Self {
            connectivity: Vec::with_capacity(triangle_count),
            normals: Directions::with_capacity(triangle_count),
            vertex_map: HashMap::with_capacity(D * triangle_count),
            vertices: Coordinates::with_capacity(D * triangle_count),
        }
    }
    fn push(&mut self, normal: [f64; D], vertices: [[f64; D]; 3]) {
        let nodes = [
            self.dedup(vertices[0]),
            self.dedup(vertices[1]),
            self.dedup(vertices[2]),
        ];
        self.connectivity.push(nodes);
        self.normals.push(Direction::const_from(normal));
    }
    fn dedup(&mut self, vertex: [f64; D]) -> usize {
        let key = vertex.map(|entry| entry.to_bits());
        if let Some(&index) = self.vertex_map.get(&key) {
            index
        } else {
            let index = self.vertices.len();
            self.vertices.push(Coordinate::const_from(vertex));
            self.vertex_map.insert(key, index);
            index
        }
    }
}

impl From<Builder> for Tessellation {
    fn from(builder: Builder) -> Self {
        let Builder {
            connectivity,
            normals,
            vertices,
            ..
        } = builder;
        Self {
            mesh: (
                vec![Connectivity::Triangular(connectivity.into())],
                vertices,
            )
                .into(),
            normals: vec![normals].into(),
            bvh: OnceCell::new(),
            features: OnceCell::new(),
        }
    }
}
