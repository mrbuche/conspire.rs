#[cfg(test)]
mod test;

use crate::{geometry::ntree::Orthotree, math::Scalar};
use std::{
    array::from_fn,
    fs::File,
    io::{BufWriter, Error, ErrorKind, Result, Write},
    path::Path,
};

pub trait WriteHtg<P>
where
    P: AsRef<Path>,
{
    fn write_htg(&self, output: P) -> Result<()>;
}

pub trait HtgValue: Copy {
    fn to_scalar(self) -> Option<Scalar>;
}

impl HtgValue for () {
    fn to_scalar(self) -> Option<Scalar> {
        None
    }
}

macro_rules! htg_value {
    ($($type:ty),*) => {
        $(impl HtgValue for $type {
            fn to_scalar(self) -> Option<Scalar> {
                Some(self as Scalar)
            }
        })*
    };
}
htg_value!(u8, u16, u32, u64, i8, i16, i32, i64, usize, f32, f64);

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V, P> WriteHtg<P>
    for Orthotree<D, L, M, N, T, U, V>
where
    P: AsRef<Path>,
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
    V: HtgValue,
{
    fn write_htg(&self, output: P) -> Result<()> {
        if D != 2 && D != 3 {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "HyperTreeGrid export supports only 2D (quadtree) or 3D (octree) trees",
            ));
        }
        let root = &self.nodes[0];
        let corner: [Scalar; D] = from_fn(|axis| root.corner[axis].into());
        let length: Scalar = root.length.into();
        let lo = self.rescale().apply(&corner.into());
        let hi = self
            .rescale()
            .apply(&from_fn(|axis| corner[axis] + length).into());
        let mut descriptor: Vec<u8> = Vec::new();
        let mut vertices_by_level: Vec<i64> = Vec::new();
        let mut depths: Vec<i64> = Vec::new();
        let mut values: Vec<Option<Scalar>> = Vec::new();
        let mut level = vec![0usize];
        let mut depth = 0;
        while !level.is_empty() {
            vertices_by_level.push(level.len() as i64);
            let mut next = Vec::new();
            for &index in &level {
                depths.push(depth);
                values.push(self.nodes[index].value.and_then(HtgValue::to_scalar));
                if let Some(orthants) = self.nodes[index].orthants() {
                    descriptor.push(1);
                    next.extend(orthants.iter().map(|&child| child.into()));
                } else {
                    descriptor.push(0);
                }
            }
            level = next;
            depth += 1;
        }
        let deepest = *vertices_by_level.last().unwrap_or(&0) as usize;
        descriptor.truncate(descriptor.len() - deepest);
        let mut file = BufWriter::new(File::create(output)?);
        writeln!(file, "<?xml version=\"1.0\"?>")?;
        writeln!(
            file,
            "<VTKFile type=\"HyperTreeGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">"
        )?;
        let dimensions: [usize; 3] = from_fn(|axis| if axis < D { 2 } else { 1 });
        writeln!(
            file,
            "  <HyperTreeGrid BranchFactor=\"2\" TransposedRootIndexing=\"0\" Dimensions=\"{} {} {}\">",
            dimensions[0], dimensions[1], dimensions[2]
        )?;
        writeln!(file, "    <Grid>")?;
        for (axis, name) in ["XCoordinates", "YCoordinates", "ZCoordinates"]
            .iter()
            .enumerate()
        {
            let values = if axis < D {
                vec![lo[axis], hi[axis]]
            } else {
                vec![0.0]
            };
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            data_array(&mut file, 6, "Float64", name, values.len(), &bytes)?;
        }
        writeln!(file, "    </Grid>")?;
        writeln!(file, "    <Trees>")?;
        writeln!(
            file,
            "      <Tree Index=\"0\" NumberOfLevels=\"{}\" NumberOfVertices=\"{}\">",
            vertices_by_level.len(),
            depths.len()
        )?;
        data_array(
            &mut file,
            8,
            "Bit",
            "Descriptor",
            descriptor.len(),
            &pack_bits(&descriptor),
        )?;
        data_array(
            &mut file,
            8,
            "Int64",
            "NbVerticesByLevel",
            vertices_by_level.len(),
            &le_i64(&vertices_by_level),
        )?;
        writeln!(file, "        <CellData>")?;
        data_array(
            &mut file,
            10,
            "Int64",
            "Depth",
            depths.len(),
            &le_i64(&depths),
        )?;
        if values.iter().any(Option::is_some) {
            let bytes: Vec<u8> = values
                .iter()
                .map(|value| value.unwrap_or(Scalar::NAN))
                .flat_map(Scalar::to_le_bytes)
                .collect();
            data_array(&mut file, 10, "Float64", "Value", values.len(), &bytes)?;
        }
        writeln!(file, "        </CellData>")?;
        writeln!(file, "      </Tree>")?;
        writeln!(file, "    </Trees>")?;
        writeln!(file, "  </HyperTreeGrid>")?;
        writeln!(file, "</VTKFile>")?;
        Ok(())
    }
}

fn data_array<W: Write>(
    file: &mut W,
    indent: usize,
    data_type: &str,
    name: &str,
    tuples: usize,
    data: &[u8],
) -> Result<()> {
    writeln!(
        file,
        "{:indent$}<DataArray type=\"{data_type}\" Name=\"{name}\" NumberOfTuples=\"{tuples}\" format=\"binary\">{}</DataArray>",
        "",
        payload(data),
    )
}

fn le_i64(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn pack_bits(bits: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0u8; bits.len().div_ceil(8)];
    for (i, &bit) in bits.iter().enumerate() {
        if bit != 0 {
            bytes[i / 8] |= 1 << (7 - i % 8);
        }
    }
    bytes
}

fn payload(data: &[u8]) -> String {
    let mut buffer = Vec::with_capacity(8 + data.len());
    buffer.extend_from_slice(&(data.len() as u64).to_le_bytes());
    buffer.extend_from_slice(data);
    base64(&buffer)
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let triple = ((chunk[0] as u32) << 16)
            | ((*chunk.get(1).unwrap_or(&0) as u32) << 8)
            | (*chunk.get(2).unwrap_or(&0) as u32);
        out.push(ALPHABET[(triple >> 18 & 63) as usize] as char);
        out.push(ALPHABET[(triple >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(triple >> 6 & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(triple & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}
