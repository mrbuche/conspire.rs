#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivities, Connectivity, Mesh, NodeSet, SideSet},
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

pub(crate) trait ReadAbaqus<P>
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
        let mut node_labels = Vec::<usize>::new();
        let mut node_index = HashMap::<usize, usize>::new();
        let mut element_index = HashMap::<usize, usize>::new();
        let mut blocks = Vec::<Connectivity>::new();
        let mut node_sets = Vec::<NodeSet>::new();
        let mut side_sets = Vec::<SideSet>::new();
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
                        if node_index.insert(id, points.len()).is_some() {
                            return Err(invalid(format!("*Node repeats node {id}")));
                        }
                        node_labels.push(id);
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
                    let nodes = tokens(&mut lines)
                        .into_iter()
                        .map(|token| {
                            let id = parse(token)?;
                            node_index.get(&id).copied().ok_or_else(|| {
                                invalid(format!("*Nset references unknown node {id}"))
                            })
                        })
                        .collect::<Result<_>>()?;
                    let node_set = NodeSet::new(nodes);
                    node_sets.push(match parameter(&parameters, "NSET") {
                        Some(name) => node_set.with_name(name),
                        None => node_set,
                    });
                }
                "SURFACE" => {
                    let sides = records(&mut lines, 2)?
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
                        .collect::<Result<_>>()?;
                    let side_set = SideSet::new(sides);
                    side_sets.push(match parameter(&parameters, "NAME") {
                        Some(name) => side_set.with_name(name),
                        None => side_set,
                    });
                }
                _ => {}
            }
        }
        let coordinates: Coordinates<D> = points.into_iter().map(|point| point.into()).collect();
        let mut mesh = Mesh::<D>::from((Connectivities::from(blocks), coordinates, node_labels));
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
    let (mut block, labels) = match element_type {
        "S3" | "S3R" | "CPS3" | "CPE3" | "STRI3" => {
            let (nodes, labels) = elements::<3>(lines, node_index, element_index, offset)?;
            (Connectivity::Triangular(nodes.into()), labels)
        }
        "S4" | "S4R" | "CPS4" | "CPS4R" | "CPE4" => {
            let (nodes, labels) = elements::<4>(lines, node_index, element_index, offset)?;
            (Connectivity::Quadrilateral(nodes.into()), labels)
        }
        "C3D4" => {
            let (nodes, labels) = elements::<4>(lines, node_index, element_index, offset)?;
            (Connectivity::Tetrahedral(nodes.into()), labels)
        }
        "C3D5" => {
            let (nodes, labels) = elements::<5>(lines, node_index, element_index, offset)?;
            (Connectivity::Pyramidal(nodes.into()), labels)
        }
        "C3D6" => {
            let (nodes, labels) = elements::<6>(lines, node_index, element_index, offset)?;
            (Connectivity::Wedge(nodes.into()), labels)
        }
        "C3D8" | "C3D8R" => {
            let (nodes, labels) = elements::<8>(lines, node_index, element_index, offset)?;
            (Connectivity::Hexahedral(nodes.into()), labels)
        }
        other => return Err(invalid(format!("unsupported Abaqus element type: {other}"))),
    };
    block.label_elements(labels);
    Ok(block)
}

fn elements<const N: usize>(
    lines: &mut Peekable<Lines>,
    node_index: &HashMap<usize, usize>,
    element_index: &mut HashMap<usize, usize>,
    offset: usize,
) -> Result<(Vec<[usize; N]>, Vec<usize>)> {
    records(lines, 1 + N)?
        .into_iter()
        .enumerate()
        .map(|(local, chunk)| {
            let label = parse(chunk[0])?;
            if element_index.insert(label, offset + local).is_some() {
                return Err(invalid(format!("*Element repeats element {label}")));
            }
            let mut nodes = [0; N];
            for (node, token) in nodes.iter_mut().zip(&chunk[1..]) {
                let id = parse(token)?;
                *node = *node_index
                    .get(&id)
                    .ok_or_else(|| invalid(format!("element references unknown node {id}")))?;
            }
            Ok((nodes, label))
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
