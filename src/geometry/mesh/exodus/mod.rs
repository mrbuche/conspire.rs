use crate::{
    geometry::mesh::PrimitiveMesh,
    io::{DefineVariable, NetCDF, PutVariable},
};
use std::{ffi::NulError, path::Path};

pub trait Write<P>
where
    P: AsRef<Path>,
{
    fn write(self, output: P) -> Result<NetCDF, NulError>;
}

// impl<const D: usize, const I: usize, const M: usize, const N: usize, P, T> Write<P>
//     for PrimitiveMesh<D, I, M, N, T>
// where
//     P: AsRef<Path>,
// {
//     fn write(self, output: P) -> Result<NetCDF, NulError> {
//         todo!()
//     }
// }

impl<const D: usize, const I: usize, const M: usize, const N: usize, P, T> Write<P>
    for &PrimitiveMesh<D, I, M, N, T>
where
    P: AsRef<Path>,
    T: Copy + Into<usize>,
{
    fn write(self, output: P) -> Result<NetCDF, NulError> {
        let path = output.as_ref().to_str().unwrap();
        let mut netcdf = NetCDF::create(path)?;
        netcdf.global()?;
        netcdf.define_dimension("num_dim", D)?;
        netcdf.define_dimension("num_elem", self.connectivity().len())?;
        netcdf.define_dimension("num_el_blk", 1)?;
        netcdf.define_dimension("num_el_in_blk1", self.connectivity().len())?;
        netcdf.define_dimension("num_nod_per_el1", N)?;
        netcdf.define_dimension("num_nodes", self.number_of_nodes())?;
        netcdf.define_dimension("time_step", 0)?;
        netcdf.define_variable::<i32>("eb_prop1", 1, &["num_el_blk"])?;
        netcdf.define_variable::<i32>("connect1", 2, &["num_el_in_blk1", "num_nod_per_el1"])?;
        let element_type = match [D, M, N] {
            [3, 2, 3] => "TRI3",
            _ => unimplemented!(),
        };
        netcdf.put_variable_attribute_text("eb_prop1", "name", "ID")?;
        netcdf.put_variable_attribute_text("connect1", "elem_type", element_type)?;
        match D {
            3 => {
                netcdf.define_variable::<f64>("coordx", 1, &["num_nodes"])?;
                netcdf.define_variable::<f64>("coordy", 1, &["num_nodes"])?;
                netcdf.define_variable::<f64>("coordz", 1, &["num_nodes"])?;
            }
            _ => unimplemented!(),
        }
        netcdf.end_definition();
        netcdf.put_variable("eb_prop1", &[1])?;
        netcdf.put_variable(
            "connect1",
            &self
                .connectivity()
                .iter()
                .flat_map(|nodes| nodes.iter().map(|&node| node.into() as i32 + 1))
                .collect::<Vec<_>>(),
        )?;
        match D {
            3 => {
                let coordinates = <[Vec<f64>; D]>::from(self.coordinates());
                netcdf.put_variable("coordx", &coordinates[0])?;
                netcdf.put_variable("coordy", &coordinates[1])?;
                netcdf.put_variable("coordz", &coordinates[2])?;
            }
            _ => unimplemented!(),
        }
        Ok(netcdf)
    }
}
