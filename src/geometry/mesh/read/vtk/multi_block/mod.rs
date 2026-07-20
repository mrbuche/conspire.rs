#[cfg(test)]
mod test;

use super::unstructured::{
    Encoding, ReadVtkUnstructured, attribute, data_array, integers, invalid, region, tag,
    unsupported,
};
use crate::geometry::mesh::Mesh;
use std::{
    fs::read_to_string,
    io::Result,
    path::{Path, PathBuf},
};

pub trait ReadVtkMultiBlock<P>
where
    P: AsRef<Path>,
    Self: Sized,
{
    fn read_vtk_multi_block(input: P) -> Result<Self>;
}

impl<const D: usize, P> ReadVtkMultiBlock<P> for Mesh<D>
where
    P: AsRef<Path>,
{
    fn read_vtk_multi_block(input: P) -> Result<Self> {
        let path = input.as_ref();
        let dir = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty());
        let join = |name: &str| -> PathBuf {
            match dir {
                Some(dir) => dir.join(name),
                None => PathBuf::from(name),
            }
        };
        let text = read_to_string(path)?;
        let header = tag(&text, "<VTKFile")?;
        if attribute(header, "type") != Some("vtkMultiBlockDataSet") {
            return Err(invalid("file is not a vtkMultiBlockDataSet".into()));
        }
        let entries = data_sets(&text)?;
        let volume_file = entries
            .iter()
            .find_map(|(name, file)| (name == "volume").then_some(file))
            .ok_or_else(|| invalid("vtkMultiBlockDataSet has no \"volume\" block".into()))?;
        let mut mesh = Mesh::<D>::read_vtk_unstructured(join(volume_file))?;
        let mut side_sets = Vec::new();
        let mut side_set_labels = Vec::new();
        for (name, file) in &entries {
            if let Some(label) = name.strip_prefix("side_set_") {
                side_sets.push(read_side_set(&join(file))?);
                side_set_labels.push(label);
            }
        }
        if !side_sets.is_empty() {
            let numbers: Option<Vec<usize>> = side_set_labels
                .iter()
                .map(|label| label.parse().ok())
                .collect();
            mesh.set_side_sets(match numbers {
                Some(numbers) => (side_sets, numbers).into(),
                None => side_sets.into(),
            });
        }
        Ok(mesh)
    }
}

fn data_sets(text: &str) -> Result<Vec<(String, String)>> {
    let mut rest = text;
    let mut entries = Vec::new();
    while let Some(start) = rest.find("<DataSet") {
        let end = rest[start..]
            .find('>')
            .ok_or_else(|| invalid("unterminated DataSet in vtkMultiBlockDataSet".into()))?
            + start;
        let tag = &rest[start..end];
        let name = attribute(tag, "name")
            .ok_or_else(|| invalid("DataSet without a name".into()))?
            .to_string();
        let file = attribute(tag, "file")
            .ok_or_else(|| invalid("DataSet without a file".into()))?
            .to_string();
        entries.push((name, file));
        rest = &rest[end..];
    }
    Ok(entries)
}

fn read_side_set(path: &Path) -> Result<Vec<(usize, usize)>> {
    let text = read_to_string(path)?;
    let header = tag(&text, "<VTKFile")?;
    if attribute(header, "type") != Some("PolyData") {
        return Err(invalid("side set file is not a PolyData".into()));
    }
    let compressor = attribute(header, "compressor");
    if matches!(compressor, Some(other) if other != "vtkZLibDataCompressor") {
        return Err(unsupported(
            "only the vtkZLibDataCompressor VTU compressor is supported",
        ));
    }
    let encoding = Encoding {
        header_bytes: match attribute(header, "header_type") {
            Some("UInt32") | None => 4,
            Some("UInt64") => 8,
            Some(other) => return Err(invalid(format!("unsupported header_type {other}"))),
        },
        compressed: compressor.is_some(),
    };
    let cell_data = region(&text, "CellData")?;
    let elements = integers(
        &data_array(cell_data, Some("OriginalElementIds"))?,
        &encoding,
    )?;
    let ordinals = integers(&data_array(cell_data, Some("OriginalFaceIds"))?, &encoding)?;
    Ok(elements
        .into_iter()
        .zip(ordinals)
        .map(|(element, ordinal)| (element as usize, ordinal as usize))
        .collect())
}
