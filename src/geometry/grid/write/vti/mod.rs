use crate::{
    geometry::grid::Grid,
    io::{
        NpyType,
        write::{data_array, data_array_compressed},
    },
};
use std::{
    array::from_fn,
    fs::File,
    io::{BufWriter, Error, ErrorKind, Result, Write},
    path::Path,
};

pub enum Vti<P>
where
    P: AsRef<Path>,
{
    Compressed(P),
    Uncompressed(P),
}

impl<P> AsRef<Path> for Vti<P>
where
    P: AsRef<Path>,
{
    fn as_ref(&self) -> &Path {
        match self {
            Vti::Compressed(path) => path.as_ref(),
            Vti::Uncompressed(path) => path.as_ref(),
        }
    }
}

pub(super) fn write<const D: usize, T, P>(
    voxels: &Grid<D, T>,
    path: P,
    compress: bool,
) -> Result<()>
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
    for &value in voxels.data_col_major().iter() {
        value.write_le(&mut data);
    }
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(file, "<?xml version=\"1.0\"?>")?;
    if compress {
        writeln!(
            file,
            "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\" compressor=\"vtkZLibDataCompressor\">"
        )?;
    } else {
        writeln!(
            file,
            "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">"
        )?;
    }
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
        if compress {
            data_array_compressed(&data)
        } else {
            data_array(&data)
        }
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
