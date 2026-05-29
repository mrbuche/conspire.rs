#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh},
    },
    io::{GetVariable, NetCDF},
};
use std::{array::from_fn, ffi::NulError, path::Path};

pub trait ReadExodus<P>
where
    P: AsRef<Path>,
    Self: Sized,
{
    fn read_exodus(input: P) -> Result<Self, NulError>;
}

impl<const D: usize, P> ReadExodus<P> for Mesh<D>
where
    P: AsRef<Path>,
{
    fn read_exodus(input: P) -> Result<Self, NulError> {
        let path = input.as_ref().to_str().unwrap();
        let netcdf = NetCDF::open(path)?;

        let num_dim = netcdf.dimension_length("num_dim")?;
        assert_eq!(
            num_dim, D,
            "exodus num_dim={num_dim} but Mesh was asked for D={D}"
        );
        let num_el_blk = netcdf.dimension_length("num_el_blk")?;
        let num_nodes = netcdf.dimension_length("num_nodes")?;

        let mut connectivities = Vec::with_capacity(num_el_blk);
        for block in 1..=num_el_blk {
            connectivities.push(read_block(&netcdf, D, block)?);
        }

        let coordx = netcdf.get_variable::<f64>("coordx", num_nodes)?;
        let coordy = netcdf.get_variable::<f64>("coordy", num_nodes)?;
        let coordz = match D {
            2 => Vec::new(),
            3 => netcdf.get_variable::<f64>("coordz", num_nodes)?,
            _ => unimplemented!(),
        };
        let coordinates: Coordinates<D> = (0..num_nodes)
            .map(|i| -> [f64; D] {
                from_fn(|ax| match ax {
                    0 => coordx[i],
                    1 => coordy[i],
                    2 => coordz[i],
                    _ => unreachable!(),
                })
            })
            .collect::<Vec<[f64; D]>>()
            .into();

        Ok(Mesh::from((connectivities, coordinates)))
    }
}

/// Read one element block from an open exodus file. Currently primitive only
/// — `connect{block}` plus its `elem_type` attribute disambiguates the variant.
fn read_block(netcdf: &NetCDF, d: usize, block: usize) -> Result<Connectivity, NulError> {
    let num_el_in_blk = netcdf.dimension_length(&format!("num_el_in_blk{}", block))?;
    let num_nod_per_el = netcdf
        .try_dimension_length(&format!("num_nod_per_el{}", block))?
        .expect("polytopal blocks not yet supported by ReadExodus");
    let elem_type = netcdf
        .get_variable_attribute_text(&format!("connect{}", block), "elem_type")?
        .to_lowercase();
    let flat = netcdf.get_variable::<i32>(
        &format!("connect{}", block),
        num_el_in_blk * num_nod_per_el,
    )?;
    match (d, num_nod_per_el, elem_type.as_str()) {
        (3, 8, _) => Ok(Connectivity::Hexahedral(unflatten::<8>(&flat).into())),
        (3, 4, "tet4") => Ok(Connectivity::Tetrahedral(unflatten::<4>(&flat).into())),
        (3, 4, "quad4") => Ok(Connectivity::Quadrilateral(unflatten::<4>(&flat).into())),
        (2, 4, _) => Ok(Connectivity::Quadrilateral(unflatten::<4>(&flat).into())),
        (_, 3, _) => Ok(Connectivity::Triangular(unflatten::<3>(&flat).into())),
        _ => panic!("unknown element type: D={d}, N={num_nod_per_el}, elem_type={elem_type}"),
    }
}

/// `connect{block}` is stored 1-indexed and flattened — split into per-element
/// `[usize; N]` arrays, dropping the +1 offset.
fn unflatten<const N: usize>(flat: &[i32]) -> Vec<[usize; N]> {
    assert_eq!(flat.len() % N, 0);
    flat.chunks_exact(N)
        .map(|chunk| from_fn(|i| (chunk[i] - 1) as usize))
        .collect()
}