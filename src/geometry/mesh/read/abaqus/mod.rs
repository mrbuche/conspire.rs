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
    collections::HashMap,
    fs::read_to_string,
    io::{Error, ErrorKind, Result},
    iter::Peekable,
    path::Path,
    str::{FromStr, Lines},
};

pub trait ReadAbaqus<P>
where
    P: AsRef<Path>,
    Self: Sized,
{
    fn read_abaqus(input: P) -> Result<Self>;
}

impl<const D: usize, P> ReadAbaqus<P> for Mesh<D>
where
    P: AsRef<Path>,
{
    fn read_abaqus(input: P) -> Result<Self> {
        let text = read_to_string(input)?;
        let mut lines = text.lines().peekable();
        let mut points = Vec::<[Scalar; D]>::new();
        let mut node_index = HashMap::<usize, usize>::new();
        let mut element_index = HashMap::<usize, usize>::new();
        let mut blocks = Vec::<Connectivity>::new();
        let mut node_sets = Vec::<Vec<usize>>::new();
        let mut side_sets = Vec::<Vec<(usize, usize)>>::new();
        while let Some(line) = lines.next() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("**") || !line.starts_with('*') {
                continue;
            }
            let parameters: Vec<&str> = line[1..].split(',').map(str::trim).collect();
            let keyword = parameters[0].to_uppercase();
            match keyword.as_str() {
                "NODE" => {
                    for chunk in records(&mut lines, 1 + D)? {
                        let id = parse(chunk[0])?;
                        let mut point = [0.0; D];
                        for (value, token) in point.iter_mut().zip(&chunk[1..]) {
                            *value = parse(token)?;
                        }
                        node_index.insert(id, points.len());
                        points.push(point);
                    }
                }
                "ELEMENT" => {
                    let element_type = parameter(&parameters, "TYPE")
                        .ok_or_else(|| invalid("*Element without a type".into()))?
                        .to_uppercase();
                    let offset = blocks.iter().map(Connectivity::number_of_elements).sum();
                    blocks.push(element_block(
                        &element_type,
                        &mut lines,
                        &node_index,
                        &mut element_index,
                        offset,
                    )?);
                }
                "NSET" => {
                    node_sets.push(
                        tokens(&mut lines)
                            .into_iter()
                            .map(|token| {
                                let id = parse(token)?;
                                node_index.get(&id).copied().ok_or_else(|| {
                                    invalid(format!("*Nset references unknown node {id}"))
                                })
                            })
                            .collect::<Result<_>>()?,
                    );
                }
                "SURFACE" => {
                    side_sets.push(
                        records(&mut lines, 2)?
                            .into_iter()
                            .map(|chunk| {
                                let label = parse(chunk[0])?;
                                let element = element_index.get(&label).copied().ok_or_else(|| {
                                    invalid(format!("*Surface references unknown element {label}"))
                                })?;
                                let side = side_label(chunk[1])?;
                                let connectivity = element_connectivity(&blocks, element)?;
                                let ordinal = (0..connectivity.local_faces().len())
                                    .find(|&ordinal| connectivity.abaqus_side(ordinal) == side)
                                    .ok_or_else(|| {
                                        invalid(format!(
                                            "unknown Abaqus side label S{side} for this element type"
                                        ))
                                    })?;
                                Ok((element, ordinal))
                            })
                            .collect::<Result<_>>()?,
                    );
                }
                _ => {}
            }
        }
        let coordinates: Coordinates<D> = points.into_iter().map(|point| point.into()).collect();
        let mut mesh = Mesh::<D>::from((blocks, coordinates));
        if !node_sets.is_empty() {
            mesh.set_node_sets(node_sets.into());
        }
        if !side_sets.is_empty() {
            mesh.set_side_sets(side_sets.into());
        }
        Ok(mesh)
    }
}

fn element_block(
    element_type: &str,
    lines: &mut Peekable<Lines>,
    node_index: &HashMap<usize, usize>,
    element_index: &mut HashMap<usize, usize>,
    offset: usize,
) -> Result<Connectivity> {
    Ok(match element_type {
        "S3" | "S3R" | "CPS3" | "CPE3" | "STRI3" => Connectivity::Triangular(
            elements::<3>(lines, node_index, element_index, offset)?.into(),
        ),
        "S4" | "S4R" | "CPS4" | "CPS4R" | "CPE4" => Connectivity::Quadrilateral(
            elements::<4>(lines, node_index, element_index, offset)?.into(),
        ),
        "C3D4" => Connectivity::Tetrahedral(
            elements::<4>(lines, node_index, element_index, offset)?.into(),
        ),
        "C3D5" => {
            Connectivity::Pyramidal(elements::<5>(lines, node_index, element_index, offset)?.into())
        }
        "C3D6" => {
            Connectivity::Wedge(elements::<6>(lines, node_index, element_index, offset)?.into())
        }
        "C3D8" | "C3D8R" => Connectivity::Hexahedral(
            elements::<8>(lines, node_index, element_index, offset)?.into(),
        ),
        other => return Err(invalid(format!("unsupported Abaqus element type: {other}"))),
    })
}

fn elements<const N: usize>(
    lines: &mut Peekable<Lines>,
    node_index: &HashMap<usize, usize>,
    element_index: &mut HashMap<usize, usize>,
    offset: usize,
) -> Result<Vec<[usize; N]>> {
    records(lines, 1 + N)?
        .into_iter()
        .enumerate()
        .map(|(local, chunk)| {
            let label = parse(chunk[0])?;
            element_index.insert(label, offset + local);
            let mut nodes = [0; N];
            for (node, token) in nodes.iter_mut().zip(&chunk[1..]) {
                let id = parse(token)?;
                *node = *node_index
                    .get(&id)
                    .ok_or_else(|| invalid(format!("element references unknown node {id}")))?;
            }
            Ok(nodes)
        })
        .collect()
}

fn element_connectivity(blocks: &[Connectivity], element: usize) -> Result<&Connectivity> {
    let mut offset = 0;
    for connectivity in blocks {
        let count = connectivity.number_of_elements();
        if element < offset + count {
            return Ok(connectivity);
        }
        offset += count;
    }
    Err(invalid(format!("element index {element} out of range")))
}

fn side_label(token: &str) -> Result<usize> {
    parse(token.trim().trim_start_matches(['S', 's']))
}

fn tokens<'a>(lines: &mut Peekable<Lines<'a>>) -> Vec<&'a str> {
    let mut tokens = Vec::new();
    while let Some(line) = lines.peek() {
        let trimmed = line.trim();
        if trimmed.starts_with('*') {
            break;
        }
        lines.next();
        if trimmed.is_empty() || trimmed.starts_with("**") {
            continue;
        }
        tokens.extend(
            trimmed
                .split(',')
                .map(str::trim)
                .filter(|token| !token.is_empty()),
        );
    }
    tokens
}

fn records<'a>(lines: &mut Peekable<Lines<'a>>, per_record: usize) -> Result<Vec<Vec<&'a str>>> {
    let data = tokens(lines);
    if !data.len().is_multiple_of(per_record) {
        return Err(invalid(format!(
            "{} data tokens not divisible by record size {per_record}",
            data.len()
        )));
    }
    Ok(data.chunks(per_record).map(<[&str]>::to_vec).collect())
}

fn parameter<'a>(parameters: &[&'a str], key: &str) -> Option<&'a str> {
    parameters.iter().find_map(|parameter| {
        let mut split = parameter.splitn(2, '=');
        let found = split.next()?.trim();
        found.eq_ignore_ascii_case(key).then(|| split.next())?
    })
}

fn parse<T>(token: &str) -> Result<T>
where
    T: FromStr,
{
    token
        .parse()
        .map_err(|_| invalid(format!("could not parse '{token}' in Abaqus file")))
}

fn invalid(message: String) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}
