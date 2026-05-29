#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::Mesh,
    io::{DefineVariable, NetCDF, PutVariable},
};
use std::{ffi::NulError, path::Path};

pub trait WriteExodus<P>
where
    P: AsRef<Path>,
{
    fn write_exodus(&self, output: P) -> Result<(), NulError>;
}

impl<const D: usize, P> WriteExodus<P> for Mesh<D>
where
    P: AsRef<Path>,
{
    fn write_exodus(&self, output: P) -> Result<(), NulError> {
        let path = output.as_ref().to_str().unwrap();
        let mut netcdf = NetCDF::create(path)?;
        netcdf.global()?;
        netcdf.define_dimension("num_dim", D)?;
        netcdf.define_dimension("num_elem", self.number_of_elements())?;
        netcdf.define_dimension("num_el_blk", self.number_of_blocks())?;
        netcdf.define_variable::<i32>("eb_prop1", 1, &["num_el_blk"])?;
        netcdf.put_variable_attribute_text("eb_prop1", "name", "ID")?;
        self.connectivities()
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
        self.connectivities()
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
        let coordinates: [Vec<f64>; D] = self.coordinates().into();
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
