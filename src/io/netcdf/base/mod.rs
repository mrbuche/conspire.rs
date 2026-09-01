use crate::io::netcdf::{
    NetCDF, Output, Reader, State, Writer,
    format::{self, AttValue, Attribute, DimSpec},
    nc_lock, reject_nul,
};
use std::{collections::HashMap, ffi::NulError, fs::File, io::Write};

impl NetCDF {
    pub fn close(&mut self) {
        let _guard = nc_lock();
        if let State::Write(writer) = &mut self.state
            && let Some(output) = &mut writer.output
        {
            let _ = output.file.flush();
        }
    }
    pub fn create(path: &str) -> Result<Self, NulError> {
        reject_nul(path)?;
        Ok(Self {
            state: State::Write(Writer {
                path: path.to_string(),
                dims: Vec::new(),
                global_attributes: Vec::new(),
                variables: Vec::new(),
                output: None,
            }),
        })
    }
    pub fn open(path: &str) -> Result<Self, NulError> {
        reject_nul(path)?;
        let _guard = nc_lock();
        let bytes = std::fs::read(path).expect("failed to read netCDF file");
        let parsed = format::parse(&bytes);
        Ok(Self {
            state: State::Read(Reader { bytes, parsed }),
        })
    }
    pub fn dimension_length(&self, name: &str) -> Result<usize, NulError> {
        reject_nul(name)?;
        Ok(self
            .lookup_dimension(name)
            .unwrap_or_else(|| panic!("no dimension named {name}")) as usize)
    }
    pub fn try_dimension_length(&self, name: &str) -> Result<Option<usize>, NulError> {
        reject_nul(name)?;
        Ok(self.lookup_dimension(name).map(|len| len as usize))
    }
    fn lookup_dimension(&self, name: &str) -> Option<u64> {
        let dims: &[DimSpec] = match &self.state {
            State::Read(reader) => &reader.parsed.dims,
            State::Write(writer) => &writer.dims,
        };
        dims.iter().find(|dim| dim.name == name).map(|dim| dim.len)
    }
    pub fn get_variable_attribute_text(
        &self,
        variable: &str,
        attr_name: &str,
    ) -> Result<String, NulError> {
        reject_nul(variable)?;
        reject_nul(attr_name)?;
        let attributes: &[Attribute] = match &self.state {
            State::Read(reader) => reader
                .parsed
                .vars
                .iter()
                .find(|var| var.name == variable)
                .map(|var| var.atts.as_slice()),
            State::Write(writer) => writer
                .variables
                .iter()
                .find(|var| var.name == variable)
                .map(|var| var.attributes.as_slice()),
        }
        .unwrap_or_else(|| panic!("no variable named {variable}"));
        match attributes
            .iter()
            .find(|attribute| attribute.name == attr_name)
            .map(|attribute| &attribute.value)
        {
            Some(AttValue::Text(text)) => Ok(text.clone()),
            _ => panic!("no text attribute {variable}::{attr_name}"),
        }
    }
    pub fn define_dimension(&mut self, name: &str, len: usize) -> Result<(), NulError> {
        reject_nul(name)?;
        let _guard = nc_lock();
        self.writer_defining().dims.push(DimSpec {
            name: name.to_string(),
            len: len as u64,
        });
        Ok(())
    }
    pub fn end_definition(&mut self) {
        let _guard = nc_lock();
        let writer = match &mut self.state {
            State::Write(writer) => writer,
            State::Read(_) => panic!("end_definition on a NetCDF opened for reading"),
        };
        assert!(writer.output.is_none(), "end_definition called twice");
        let dim_index: HashMap<&str, usize> = writer
            .dims
            .iter()
            .enumerate()
            .map(|(index, dim)| (dim.name.as_str(), index))
            .collect();
        let mut variables: Vec<format::VarSpec> = writer
            .variables
            .iter_mut()
            .map(|build| format::VarSpec {
                name: build.name.clone(),
                xtype: build.xtype,
                dimids: build
                    .dim_names
                    .iter()
                    .map(|dim| {
                        *dim_index.get(dim.as_str()).unwrap_or_else(|| {
                            panic!("variable references unknown dimension {dim}")
                        })
                    })
                    .collect(),
                atts: std::mem::take(&mut build.attributes),
                begin: 0,
                vsize: 0,
                storage: format::Storage::Classic,
            })
            .collect();
        let header = format::finalize(&writer.dims, &writer.global_attributes, &mut variables);
        let mut file = File::create(&writer.path).expect("failed to create netCDF file");
        file.write_all(&header)
            .expect("failed to write netCDF header");
        writer.output = Some(Output { file, variables });
    }
    pub fn global(&mut self) {
        let _guard = nc_lock();
        let title = format!("{} v{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        self.writer_defining().global_attributes.extend([
            Attribute {
                name: "api_version".to_string(),
                value: AttValue::Float(vec![8.25]),
            },
            Attribute {
                name: "file_size".to_string(),
                value: AttValue::Int(vec![1]),
            },
            Attribute {
                name: "floating_point_word_size".to_string(),
                value: AttValue::Int(vec![8]),
            },
            Attribute {
                name: "version".to_string(),
                value: AttValue::Float(vec![8.25]),
            },
            Attribute {
                name: "title".to_string(),
                value: AttValue::Text(title),
            },
        ]);
    }
    pub fn put_variable_attribute_text(
        &mut self,
        variable: &str,
        attr_name: &str,
        value: &str,
    ) -> Result<(), NulError> {
        reject_nul(variable)?;
        reject_nul(attr_name)?;
        reject_nul(value)?;
        let _guard = nc_lock();
        let build = self
            .writer_defining()
            .variables
            .iter_mut()
            .find(|build| build.name == variable)
            .unwrap_or_else(|| panic!("no variable named {variable}"));
        build.attributes.push(Attribute {
            name: attr_name.to_string(),
            value: AttValue::Text(value.to_string()),
        });
        Ok(())
    }
    pub(super) fn writer_defining(&mut self) -> &mut Writer {
        match &mut self.state {
            State::Write(writer) if writer.output.is_none() => writer,
            State::Write(_) => panic!("operation not allowed after end_definition"),
            State::Read(_) => panic!("write operation on a NetCDF opened for reading"),
        }
    }
}
