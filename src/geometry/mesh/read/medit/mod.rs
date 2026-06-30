#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh},
    },
    math::Scalar,
};
use std::{
    fs::read_to_string,
    io::{Error, ErrorKind, Result},
    path::Path,
    str::FromStr,
};

pub trait ReadMedit<P>
where
    P: AsRef<Path>,
    Self: Sized,
{
    fn read_medit(input: P) -> Result<Self>;
}

impl<const D: usize, P> ReadMedit<P> for Mesh<D>
where
    P: AsRef<Path>,
{
    fn read_medit(input: P) -> Result<Self> {
        let text = read_to_string(input)?;
        let mut tokens = text.split_whitespace();
        let mut coordinates: Vec<[Scalar; D]> = Vec::new();
        let mut blocks: Vec<Connectivity> = Vec::new();
        while let Some(keyword) = tokens.next() {
            match keyword {
                "MeshVersionFormatted" | "MeshVersionUnformatted" => {
                    next_token(&mut tokens)?;
                }
                "Dimension" => {
                    let dimension: usize = next_parse(&mut tokens)?;
                    if dimension != D {
                        return Err(invalid(format!(
                            ".mesh Dimension {dimension} but Mesh was asked for D={D}"
                        )));
                    }
                }
                "Vertices" => {
                    let count: usize = next_parse(&mut tokens)?;
                    for _ in 0..count {
                        let mut point = [0.0; D];
                        for value in &mut point {
                            *value = next_parse(&mut tokens)?;
                        }
                        let _reference: i64 = next_parse(&mut tokens)?;
                        coordinates.push(point);
                    }
                }
                "Triangles" => blocks.push(Connectivity::Triangular(
                    read_elements::<3, _>(&mut tokens)?.into(),
                )),
                "Quadrilaterals" => blocks.push(Connectivity::Quadrilateral(
                    read_elements::<4, _>(&mut tokens)?.into(),
                )),
                "Tetrahedra" => blocks.push(Connectivity::Tetrahedral(
                    read_elements::<4, _>(&mut tokens)?.into(),
                )),
                "Pyramids" => blocks.push(Connectivity::Pyramidal(
                    read_elements::<5, _>(&mut tokens)?.into(),
                )),
                "Prisms" | "Pentahedra" => blocks.push(Connectivity::Wedge(
                    read_elements::<6, _>(&mut tokens)?.into(),
                )),
                "Hexahedra" => blocks.push(Connectivity::Hexahedral(
                    read_elements::<8, _>(&mut tokens)?.into(),
                )),
                "Edges" => skip_records(&mut tokens, 3)?,
                "Corners" | "Ridges" | "RequiredVertices" | "RequiredEdges" => {
                    skip_records(&mut tokens, 1)?
                }
                "End" => break,
                other => return Err(invalid(format!("unsupported .mesh keyword: {other}"))),
            }
        }
        let coordinates: Coordinates<D> =
            coordinates.into_iter().map(|point| point.into()).collect();
        Ok((blocks, coordinates).into())
    }
}

fn invalid(message: String) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

fn next_token<'a, I>(tokens: &mut I) -> Result<&'a str>
where
    I: Iterator<Item = &'a str>,
{
    tokens
        .next()
        .ok_or_else(|| Error::new(ErrorKind::UnexpectedEof, "unexpected end of .mesh file"))
}

fn next_parse<'a, I, T>(tokens: &mut I) -> Result<T>
where
    I: Iterator<Item = &'a str>,
    T: FromStr,
{
    let token = next_token(tokens)?;
    token
        .parse()
        .map_err(|_| invalid(format!("could not parse '{token}' in .mesh file")))
}

fn read_elements<'a, const N: usize, I>(tokens: &mut I) -> Result<Vec<[usize; N]>>
where
    I: Iterator<Item = &'a str>,
{
    let count: usize = next_parse(tokens)?;
    let mut elements = Vec::with_capacity(count);
    for _ in 0..count {
        let mut nodes = [0; N];
        for node in &mut nodes {
            let one_based: usize = next_parse(tokens)?;
            *node = one_based
                .checked_sub(1)
                .ok_or_else(|| invalid("node index 0 in .mesh (expected 1-based)".into()))?;
        }
        let _reference: i64 = next_parse(tokens)?;
        elements.push(nodes);
    }
    Ok(elements)
}

fn skip_records<'a, I>(tokens: &mut I, per_record: usize) -> Result<()>
where
    I: Iterator<Item = &'a str>,
{
    let count: usize = next_parse(tokens)?;
    for _ in 0..count * per_record {
        next_token(tokens)?;
    }
    Ok(())
}
