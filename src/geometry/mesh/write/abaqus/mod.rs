#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Connectivity, Mesh},
    math::Tensor,
};
use std::{
    fs::File,
    io::{BufWriter, Error, ErrorKind, Result, Write},
    path::Path,
};

pub(crate) trait WriteAbaqus<P>
where
    P: AsRef<Path>,
{
    fn write_abaqus(&self, output: P) -> Result<()>;
}

fn abaqus_type(connectivity: &Connectivity) -> Result<&'static str> {
    Ok(match connectivity {
        Connectivity::Triangular(_) => "S3",
        Connectivity::Quadrilateral(_) => "S4",
        Connectivity::Tetrahedral(_) => "C3D4",
        Connectivity::Pyramidal(_) => "C3D5",
        Connectivity::Wedge(_) => "C3D6",
        Connectivity::Hexahedral(_) => "C3D8",
        Connectivity::Polygonal(_) | Connectivity::Polyhedral(_) => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Abaqus writer does not support polygonal/polyhedral blocks",
            ));
        }
    })
}

impl<const D: usize, P> WriteAbaqus<P> for Mesh<D>
where
    P: AsRef<Path>,
{
    fn write_abaqus(&self, output: P) -> Result<()> {
        let mut file = BufWriter::new(File::create(output)?);
        writeln!(file, "*Heading")?;
        writeln!(file, " conspire mesh")?;
        writeln!(file, "*Node")?;
        let node_labels = self.node_labels();
        let node_label = |node: usize| node_labels.map_or(node + 1, |labels| labels[node]);
        let coordinates = self.coordinates();
        for node in 0..coordinates.len() {
            write!(file, "{}", node_label(node))?;
            for i in 0..D {
                write!(file, ", {}", coordinates[node][i])?;
            }
            writeln!(file)?;
        }
        let element_labels = self
            .iter()
            .map(|connectivity| connectivity.element_labels())
            .collect::<Option<Vec<_>>>()
            .map(|labels| labels.concat());
        let element_label = |element: usize| {
            element_labels
                .as_ref()
                .map_or(element + 1, |labels| labels[element])
        };
        let mut element = 0;
        for (block, connectivity) in self.iter().enumerate() {
            let element_type = abaqus_type(connectivity)?;
            writeln!(
                file,
                "*Element, type={element_type}, elset=BLOCK{}",
                block + 1
            )?;
            for nodes in connectivity.iter() {
                write!(file, "{}", element_label(element))?;
                for &node in nodes {
                    write!(file, ", {}", node_label(node))?;
                }
                writeln!(file)?;
                element += 1;
            }
        }
        for (set, node_set) in self.node_sets().iter().enumerate() {
            match node_set.name() {
                Some(name) => writeln!(file, "*Nset, nset={name}")?,
                None => writeln!(file, "*Nset, nset=NSET{}", set + 1)?,
            }
            for chunk in node_set.nodes().chunks(16) {
                let line = chunk
                    .iter()
                    .map(|&node| node_label(node).to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(file, "{line}")?;
            }
        }
        for (set, side_set) in self.side_sets().iter().enumerate() {
            match side_set.name() {
                Some(name) => writeln!(file, "*Surface, type=ELEMENT, name={name}")?,
                None => writeln!(file, "*Surface, type=ELEMENT, name=SURF{}", set + 1)?,
            }
            for &(element, ordinal) in side_set.sides() {
                let mut offset = 0;
                let connectivity = self
                    .iter()
                    .find(|connectivity| {
                        let count = connectivity.number_of_elements();
                        let found = element < offset + count;
                        offset += count;
                        found
                    })
                    .expect("side set references an out-of-range element");
                let label = connectivity.abaqus_side(ordinal);
                writeln!(file, "{}, S{label}", element_label(element))?;
            }
        }
        Ok(())
    }
}
