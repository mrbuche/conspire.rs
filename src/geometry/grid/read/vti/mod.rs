use crate::{
    geometry::grid::Grid,
    io::{
        NpyType, invalid,
        read::{attribute, data_array, decode, encoding, tag},
        unsupported,
    },
};
use std::{array::from_fn, fs::read_to_string, io::Result, path::Path};

pub(super) fn read<const D: usize, T, P>(path: P) -> Result<Grid<D, T>>
where
    T: NpyType,
    P: AsRef<Path>,
{
    if D != 2 && D != 3 {
        return Err(unsupported("VTI supports only pixels or voxels"));
    }
    let text = read_to_string(path)?;
    let file = tag(&text, "<VTKFile")?;
    if attribute(file, "type") != Some("ImageData") {
        return Err(invalid("VTI is not ImageData".into()));
    }
    if matches!(attribute(file, "byte_order"), Some(order) if order != "LittleEndian") {
        return Err(unsupported("big-endian VTI is not supported"));
    }
    let encoding = encoding(file)?;
    let image = tag(&text, "<ImageData")?;
    let bounds: Vec<i64> = attribute(image, "WholeExtent")
        .ok_or_else(|| invalid("no WholeExtent".into()))?
        .split_whitespace()
        .map(|n| n.parse().map_err(|_| invalid("bad WholeExtent".into())))
        .collect::<Result<_>>()?;
    if bounds.len() != 6 {
        return Err(invalid("WholeExtent must have 6 entries".into()));
    }
    let cells: [usize; 3] = from_fn(|axis| (bounds[2 * axis + 1] - bounds[2 * axis]) as usize);
    if cells[D..].iter().any(|&c| c > 1) {
        return Err(invalid(format!(
            "VTI extent {cells:?} exceeds {D} dimensions"
        )));
    }
    let nel: [usize; D] = from_fn(|axis| cells[axis]);
    let total: usize = nel.iter().product();
    let array = data_array(&text, None)?;
    if array.data_type != vtk_type(T::DESCR) {
        return Err(invalid(format!(
            "DataArray type {} does not match the requested scalar",
            array.data_type
        )));
    }
    let bytes = decode(&array, &encoding)?;
    let data: Vec<T> = bytes.chunks(T::SIZE).take(total).map(T::read_le).collect();
    Ok(Grid::new(data, nel))
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
