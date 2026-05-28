#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{MeshNew, PrimitiveMesh},
    io::{DefineVariable, NetCDF, PutVariable},
};
use std::{ffi::NulError, path::Path};

pub trait WriteExodus<P>
where
    P: AsRef<Path>,
{
    fn write_exodus(self, output: P) -> Result<(), NulError>;
}

struct FlatConnectivity(Vec<i32>);

impl<const N: usize, T> From<Vec<[T; N]>> for FlatConnectivity
where
    T: Copy + Into<usize>,
{
    fn from(connectivity: Vec<[T; N]>) -> Self {
        Self(
            connectivity
                .into_iter()
                .flat_map(|nodes| nodes.into_iter().map(|node| node.into() as i32 + 1))
                .collect(),
        )
    }
}

impl<const N: usize, T> From<&Vec<[T; N]>> for FlatConnectivity
where
    T: Copy + Into<usize>,
{
    fn from(connectivity: &Vec<[T; N]>) -> Self {
        Self(
            connectivity
                .iter()
                .flat_map(|nodes| nodes.iter().map(|&node| node.into() as i32 + 1))
                .collect(),
        )
    }
}

impl<const D: usize, const M: usize, const N: usize, P, T> WriteExodus<P>
    for PrimitiveMesh<D, M, N, T>
where
    P: AsRef<Path>,
    T: Copy + Into<usize>,
{
    fn write_exodus(self, output: P) -> Result<(), NulError> {
        let netcdf = write_exodus_prelude(&self, output)?;
        let (connectivity, coordinates) = self.into();
        write_exodus_ending(netcdf, connectivity.into(), coordinates.into())
    }
}

impl<const D: usize, const M: usize, const N: usize, P, T> WriteExodus<P>
    for &PrimitiveMesh<D, M, N, T>
where
    P: AsRef<Path>,
    T: Copy + Into<usize>,
{
    fn write_exodus(self, output: P) -> Result<(), NulError> {
        let netcdf = write_exodus_prelude(self, output)?;
        write_exodus_ending(
            netcdf,
            self.connectivity().into(),
            self.coordinates().into(),
        )
    }
}

fn write_exodus_prelude<const D: usize, const M: usize, const N: usize, P, T>(
    mesh: &PrimitiveMesh<D, M, N, T>,
    output: P,
) -> Result<NetCDF, NulError>
where
    P: AsRef<Path>,
    T: Copy + Into<usize>,
{
    let path = output.as_ref().to_str().unwrap();
    let mut netcdf = NetCDF::create(path)?;
    netcdf.global()?;
    netcdf.define_dimension("num_dim", D)?;
    netcdf.define_dimension("num_elem", mesh.connectivity().len())?;
    netcdf.define_dimension("num_el_blk", 1)?;
    netcdf.define_dimension("num_el_in_blk1", mesh.connectivity().len())?;
    netcdf.define_dimension("num_nod_per_el1", N)?;
    netcdf.define_dimension("num_nodes", mesh.number_of_nodes())?;
    netcdf.define_dimension("time_step", 0)?;
    netcdf.define_variable::<i32>("eb_prop1", 1, &["num_el_blk"])?;
    netcdf.define_variable::<i32>("connect1", 2, &["num_el_in_blk1", "num_nod_per_el1"])?;
    let element_type = match [D, M, N] {
        [2, 2, 3] => "TRI3",
        [2, 2, 4] => "QUAD4",
        [3, 2, 3] => "TRI3",
        [3, 2, 4] => "QUAD4",
        [3, 3, 4] => "TET4",
        [3, 3, 8] => "HEX8",
        _ => unimplemented!(),
    };
    netcdf.put_variable_attribute_text("eb_prop1", "name", "ID")?;
    netcdf.put_variable_attribute_text("connect1", "elem_type", element_type)?;
    netcdf.define_variable::<f64>("coordx", 1, &["num_nodes"])?;
    netcdf.define_variable::<f64>("coordy", 1, &["num_nodes"])?;
    match D {
        2 => {}
        3 => {
            netcdf.define_variable::<f64>("coordz", 1, &["num_nodes"])?;
        }
        _ => unimplemented!(),
    }
    netcdf.end_definition();
    Ok(netcdf)
}

fn write_exodus_ending<const D: usize>(
    mut netcdf: NetCDF,
    connectivity: FlatConnectivity,
    coordinates: [Vec<f64>; D],
) -> Result<(), NulError> {
    netcdf.put_variable("eb_prop1", &[1])?;
    netcdf.put_variable("connect1", &connectivity.0)?;
    netcdf.put_variable("coordx", &coordinates[0])?;
    netcdf.put_variable("coordy", &coordinates[1])?;
    match D {
        2 => {}
        3 => {
            netcdf.put_variable("coordz", &coordinates[2])?;
        }
        _ => unimplemented!(),
    }
    Ok(())
}

impl<const D: usize, P, T> WriteExodus<P> for MeshNew<D, T>
where
    P: AsRef<Path>,
    T: Copy + Into<i32>,
{
    fn write_exodus(self, output: P) -> Result<(), NulError> {
        let path = output.as_ref().to_str().unwrap();
        let mut netcdf = NetCDF::create(path)?;
        netcdf.global()?;
        netcdf.define_dimension("num_dim", D)?;
        netcdf.define_dimension("num_elem", self.number_of_elements())?;
        netcdf.define_dimension("num_el_blk", self.number_of_blocks())?;
        netcdf.define_variable::<i32>("eb_prop1", 1, &["num_el_blk"])?;
        netcdf.put_variable_attribute_text("eb_prop1", "name", "ID")?;
        self.connectivities
            .iter()
            .enumerate()
            .try_for_each(|(block, connectivity)| {
                let block = block + 1;
                netcdf.define_dimension(&format!("num_el_in_blk{}", block), connectivity.len())?;
                if let Some(num) = connectivity.number_of_nodes_per_element() {
                    netcdf.define_dimension(&format!("num_nod_per_el{}", block), num)?;
                    netcdf.define_variable::<i32>(
                        &format!("connect{}", block),
                        2,
                        &[
                            &format!("num_el_in_blk{}", block),
                            &format!("num_nod_per_el{}", block),
                        ],
                    )?;
                    netcdf.put_variable_attribute_text(
                        &format!("connect{}", block),
                        "elem_type",
                        connectivity.exodus_element_type(),
                    )
                } else {
                    Ok(())
                }
            })?;
        netcdf.define_dimension("num_nodes", self.number_of_nodes())?;
        netcdf.define_dimension("time_step", 0)?;
        netcdf.define_variable::<f64>("coordx", 1, &["num_nodes"])?;
        netcdf.define_variable::<f64>("coordy", 1, &["num_nodes"])?;
        match D {
            2 => {}
            3 => {
                netcdf.define_variable::<f64>("coordz", 1, &["num_nodes"])?;
            }
            _ => unimplemented!(),
        }
        netcdf.end_definition();
        let block_ids: Vec<i32> = (1..self.number_of_blocks() + 1)
            .map(|index| index as i32)
            .collect();
        netcdf.put_variable("eb_prop1", &block_ids)?;
        let coordinates: [Vec<f64>; D] = self.coordinates().into();
        self.connectivities
            .iter()
            .enumerate()
            .try_for_each(|(block, connectivity)| {
                let block = block + 1;
                if let Some(flat) = connectivity.primitive_connectivity_flattened() {
                    netcdf.put_variable(&format!("connect{}", block), &flat)
                } else {
                    Ok(())
                }
            })?;
        netcdf.put_variable("coordx", &coordinates[0])?;
        netcdf.put_variable("coordy", &coordinates[1])?;
        match D {
            2 => {}
            3 => {
                netcdf.put_variable("coordz", &coordinates[2])?;
            }
            _ => unimplemented!(),
        }
        Ok(())
    }
}
