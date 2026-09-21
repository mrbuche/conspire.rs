#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Connectivity, Mesh},
    math::Scalar,
};
use std::{
    fs::read_to_string,
    io::{Error, ErrorKind, Result},
    path::Path,
    str::FromStr,
};

/// Reads an Object File Format (`.off`) surface mesh.
///
/// The file holds the magic word `OFF` in any case, the counts of vertices,
/// faces, and edges, the vertices as `x y z`, and the faces as `n i_1 ... i_n`.
/// Face indices are 0-based.
/// The edge count is read but not used, since writers often leave it at zero.
/// Comments start with `#`, and blank lines are skipped.
/// Values after the last index of a face, such as a face color, are ignored.
///
/// Triangles and quadrilaterals become [`Connectivity::Triangular`] and
/// [`Connectivity::Quadrilateral`] blocks, in that order.
/// A file that mixes the two therefore reorders its faces by block.
/// Other face sizes, and the color, normal, texture, and higher-dimensional
/// variants of the format (`COFF`, `NOFF`, `STOFF`, `4OFF`, ...), return an error.
/// So does a coordinate that is not finite, such as `nan`, `inf`, or `1e999`.
/// So does any content after the last face, because that means a count in the
/// header is too small.
pub(crate) trait ReadOff<P>
where
    P: AsRef<Path>,
    Self: Sized,
{
    fn read_off(input: P) -> Result<Self>;
}

impl<const D: usize, P> ReadOff<P> for Mesh<D>
where
    P: AsRef<Path>,
{
    fn read_off(input: P) -> Result<Self> {
        if D != 3 {
            return Err(invalid(format!(
                ".off holds 3D points but Mesh was asked for D={D}"
            )));
        }
        let text = read_to_string(input)?;
        let mut lines = text
            .lines()
            .enumerate()
            .map(|(index, line)| (index + 1, line.split('#').next().unwrap_or("").trim()))
            .filter(|(_, line)| !line.is_empty());
        let (number, first) = lines.next().ok_or_else(end_of_file)?;
        let mut header = first.split_whitespace();
        let magic = header.next().unwrap_or("");
        if !magic.eq_ignore_ascii_case("OFF") {
            return Err(invalid(format!(
                "line {number}: unsupported .off header '{magic}' (expected OFF)"
            )));
        }
        let mut counts: Vec<&str> = header.collect();
        let mut counts_number = number;
        if counts.is_empty() {
            let (next_number, next) = lines.next().ok_or_else(end_of_file)?;
            counts_number = next_number;
            counts = next.split_whitespace().collect();
        }
        if counts.len() != 3 {
            return Err(invalid(format!(
                "line {counts_number}: expected the vertex, face, and edge counts"
            )));
        }
        let vertices: usize = parse(counts[0], counts_number)?;
        let faces: usize = parse(counts[1], counts_number)?;
        let _edges: usize = parse(counts[2], counts_number)?;
        // A vertex takes at least one byte, so the file size bounds the reservation.
        // A wrong count in the header then cannot request more than the file holds.
        let mut coordinates = Vec::<[Scalar; D]>::with_capacity(vertices.min(text.len()));
        for _ in 0..vertices {
            let (number, line) = lines.next().ok_or_else(end_of_file)?;
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.len() != 3 {
                return Err(invalid(format!(
                    "line {number}: expected 3 values for a vertex, found {}",
                    tokens.len()
                )));
            }
            let mut point = [0.0; D];
            for (value, token) in point.iter_mut().zip(&tokens) {
                let coordinate: Scalar = parse(token, number)?;
                if !coordinate.is_finite() {
                    return Err(invalid(format!(
                        "line {number}: coordinate '{token}' is not finite"
                    )));
                }
                *value = coordinate;
            }
            coordinates.push(point);
        }
        let mut triangles = Vec::<[usize; 3]>::new();
        let mut quadrilaterals = Vec::<[usize; 4]>::new();
        for _ in 0..faces {
            let (number, line) = lines.next().ok_or_else(end_of_file)?;
            let tokens: Vec<&str> = line.split_whitespace().collect();
            let size = parse(tokens[0], number)?;
            if size > tokens.len() - 1 {
                return Err(invalid(format!(
                    "line {number}: a face of {size} vertices has only {} indices",
                    tokens.len() - 1
                )));
            }
            let mut nodes = Vec::with_capacity(size);
            for token in &tokens[1..=size] {
                let node = parse(token, number)?;
                if node >= vertices {
                    return Err(invalid(format!(
                        "line {number}: vertex index {node} is out of range for {vertices} vertices"
                    )));
                }
                nodes.push(node);
            }
            match size {
                3 => triangles.push([nodes[0], nodes[1], nodes[2]]),
                4 => quadrilaterals.push([nodes[0], nodes[1], nodes[2], nodes[3]]),
                _ => {
                    return Err(invalid(format!(
                        "line {number}: unsupported face of {size} vertices (expected 3 or 4)"
                    )));
                }
            }
        }
        if let Some((number, _)) = lines.next() {
            return Err(invalid(format!(
                "line {number}: unexpected content after the last face, check the counts in the header"
            )));
        }
        let mut blocks = Vec::<Connectivity>::new();
        if !triangles.is_empty() {
            blocks.push(Connectivity::Triangular(triangles.into()));
        }
        if !quadrilaterals.is_empty() {
            blocks.push(Connectivity::Quadrilateral(quadrilaterals.into()));
        }
        Ok((blocks, coordinates.into()).into())
    }
}

fn invalid(message: String) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

fn end_of_file() -> Error {
    Error::new(ErrorKind::UnexpectedEof, "unexpected end of .off file")
}

fn parse<T>(token: &str, number: usize) -> Result<T>
where
    T: FromStr,
{
    token.parse().map_err(|_| {
        invalid(format!(
            "line {number}: could not parse '{token}' in .off file"
        ))
    })
}
