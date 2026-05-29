#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Mesh, connectivity::base::FlatConnectivity},
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
        netcdf.define_dimension("num_el_blk", self.number_of_element_blocks())?;
        netcdf.define_variable::<i32>("eb_prop1", 1, &["num_el_blk"])?;
        netcdf.put_variable_attribute_text("eb_prop1", "name", "ID")?;
        if let Some(num_fa_blk) = self.number_of_face_blocks() {
            netcdf.define_dimension("num_fa_blk", num_fa_blk)?;
            netcdf.define_variable::<i32>("fa_prop1", 1, &["num_fa_blk"])?;
            netcdf.put_variable_attribute_text("fa_prop1", "name", "ID")?;
        }
        if let Some(num_face) = self.number_of_faces() {
            netcdf.define_dimension("num_face", num_face)?;
        }
        self.connectivities()
            .iter()
            .enumerate()
            .try_for_each(|(block, connectivity)| {
                let block = block + 1;
                netcdf.define_dimension(
                    &format!("num_el_in_blk{}", block),
                    connectivity.number_of_elements(),
                )?;
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
                    if let Some(ebepecnt) = connectivity.number_of_faces_per_element::<i32>() {
                        netcdf.define_dimension(
                            &format!("num_fac_per_el{}", block),
                            ebepecnt.into_iter().map(|c| c as usize).sum(),
                        )?;
                        netcdf.define_variable::<i32>(
                            &format!("facconn{}", block),
                            1,
                            &[&format!("num_fac_per_el{}", block)],
                        )?;
                        netcdf.put_variable_attribute_text(
                            &format!("facconn{}", block),
                            "elem_type",
                            connectivity.exodus_element_type(),
                        )?;
                        netcdf.define_variable::<i32>(
                            &format!("ebepecnt{}", block),
                            1,
                            &[&format!("num_el_in_blk{}", block)],
                        )?;
                        netcdf.put_variable_attribute_text(
                            &format!("ebepecnt{}", block),
                            "entity_type1",
                            "FACE",
                        )?;
                        netcdf.put_variable_attribute_text(
                            &format!("ebepecnt{}", block),
                            "entity_type2",
                            "ELEM",
                        )?;
                    } else {
                        panic!()
                    }
                    if let Some(num_fa_in_blk) = connectivity.number_of_faces() {
                        netcdf
                            .define_dimension(&format!("num_fa_in_blk{}", block), num_fa_in_blk)?;
                    } else {
                        panic!()
                    }
                    if let Some(fbepecnt) = connectivity.number_of_nodes_per_face::<i32>() {
                        netcdf.define_dimension(
                            &format!("num_nod_per_fa{}", block),
                            fbepecnt.into_iter().map(|c| c as usize).sum(),
                        )?;
                        netcdf.define_variable::<i32>(
                            &format!("fbconn{}", block),
                            1,
                            &[&format!("num_nod_per_fa{}", block)],
                        )?;
                        netcdf.put_variable_attribute_text(
                            &format!("fbconn{}", block),
                            "elem_type",
                            "nsided",
                        )?;
                        netcdf.define_variable::<i32>(
                            &format!("fbepecnt{}", block),
                            1,
                            &[&format!("num_fa_in_blk{}", block)],
                        )?;
                        netcdf.put_variable_attribute_text(
                            &format!("fbepecnt{}", block),
                            "entity_type1",
                            "NODE",
                        )?;
                        netcdf.put_variable_attribute_text(
                            &format!("fbepecnt{}", block),
                            "entity_type2",
                            "FACE",
                        )
                    } else {
                        panic!()
                    }
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
        let block_ids: Vec<i32> = (1..self.number_of_element_blocks() + 1)
            .map(|index| index as i32)
            .collect();
        netcdf.put_variable("eb_prop1", &block_ids)?;
        if self.number_of_face_blocks().is_some() {
            netcdf.put_variable("fa_prop1", &block_ids)?;
        }
        self.connectivities()
            .iter()
            .enumerate()
            .try_for_each(|(block, connectivity)| {
                let block = block + 1;
                match connectivity.flat_connectivity::<i32>() {
                    FlatConnectivity::Primitive(flat) => {
                        netcdf.put_variable(&format!("connect{}", block), &flat)
                    }
                    FlatConnectivity::Polytopal(elements_faces, faces_nodes) => {
                        netcdf.put_variable(&format!("facconn{}", block), &elements_faces)?;
                        netcdf.put_variable(&format!("fbconn{}", block), &faces_nodes)?;
                        if let Some(ebepecnt) = connectivity.number_of_faces_per_element::<i32>() {
                            netcdf.put_variable(&format!("ebepecnt{}", block), &ebepecnt)?;
                        } else {
                            panic!()
                        }
                        if let Some(fbepecnt) = connectivity.number_of_nodes_per_face::<i32>() {
                            netcdf.put_variable(&format!("fbepecnt{}", block), &fbepecnt)
                        } else {
                            panic!()
                        }
                    }
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
