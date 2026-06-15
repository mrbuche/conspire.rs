use crate::{geometry::grid::Grid, io::NpyType};
use std::{
    array::from_fn,
    fs::File,
    io::{BufWriter, Error, ErrorKind, Result, Write},
    path::Path,
};

pub(super) fn write<const D: usize, T, P>(voxels: &Grid<D, T>, path: P) -> Result<()>
where
    T: NpyType,
    P: AsRef<Path>,
{
    if D != 2 && D != 3 {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "VTI supports only pixels or voxels",
        ));
    }
    let cells: [usize; 3] = from_fn(|axis| if axis < D { voxels.nel()[axis] } else { 0 });
    let extent = format!("0 {} 0 {} 0 {}", cells[0], cells[1], cells[2]);
    let mut data = Vec::with_capacity(voxels.len() * T::SIZE);
    for &value in voxels.data() {
        value.write_le(&mut data);
    }
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(file, "<?xml version=\"1.0\"?>")?;
    writeln!(
        file,
        "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">"
    )?;
    writeln!(
        file,
        "  <ImageData WholeExtent=\"{extent}\" Origin=\"0 0 0\" Spacing=\"1 1 1\">"
    )?;
    writeln!(file, "    <Piece Extent=\"{extent}\">")?;
    writeln!(file, "      <CellData Scalars=\"data\">")?;
    writeln!(
        file,
        "        <DataArray type=\"{}\" Name=\"data\" NumberOfComponents=\"1\" format=\"binary\">{}</DataArray>",
        vtk_type(T::DESCR),
        payload(&data)
    )?;
    writeln!(file, "      </CellData>")?;
    writeln!(file, "    </Piece>")?;
    writeln!(file, "  </ImageData>")?;
    writeln!(file, "</VTKFile>")?;
    Ok(())
}

fn vtk_type(descr: &str) -> &'static str {
    match descr {
        "|u1" => "UInt8",
        "|i1" => "Int8",
        "<u2" => "UInt16",
        "<i2" => "Int16",
        "<u4" => "UInt32",
        "<i4" => "Int32",
        "<u8" => "UInt64",
        "<i8" => "Int64",
        "<f4" => "Float32",
        "<f8" => "Float64",
        _ => "UInt8",
    }
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
