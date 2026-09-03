//! ISO 10303-21 exchange-file syntax.
//!
//! Parses the `#N = ENTITY(...)` grammar shared by every STEP application
//! protocol into an entity map. Carries no geometry or AP-specific meaning;
//! that decoding lives in the STEP reader.

#[cfg(test)]
mod test;

use crate::io::invalid;
use std::{
    collections::BTreeMap,
    io::{Error, Result},
    str::{FromStr, from_utf8},
};

#[derive(Clone, Debug, PartialEq)]
pub enum Parameter {
    Integer(i64),
    Real(f64),
    String(String),
    Enumeration(String),
    Reference(u64),
    List(Vec<Parameter>),
    Typed {
        keyword: String,
        parameter: Box<Parameter>,
    },
    Null,
    Derived,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub keyword: String,
    pub parameters: Vec<Parameter>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    pub records: Vec<Record>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Exchange {
    pub header: Vec<Record>,
    pub data: BTreeMap<u64, Instance>,
}

impl FromStr for Exchange {
    type Err = Error;
    fn from_str(text: &str) -> Result<Self> {
        parse(text)
    }
}

pub fn parse(text: &str) -> Result<Exchange> {
    // Tolerate a leading UTF-8 BOM some exporters prepend; keep its width so
    // reported byte offsets still point into the original file.
    let (text, base_offset) = match text.strip_prefix('\u{feff}') {
        Some(rest) => (rest, '\u{feff}'.len_utf8()),
        None => (text, 0),
    };
    let mut scanner = Scanner {
        bytes: text.as_bytes(),
        position: 0,
        depth: 0,
        base_offset,
        comment_unterminated: false,
    };
    scanner.literal("ISO-10303-21")?;
    scanner.terminator()?;
    scanner.literal("HEADER")?;
    scanner.terminator()?;
    let mut header = Vec::new();
    while !scanner.try_literal("ENDSEC") {
        header.push(scanner.record()?);
        scanner.terminator()?;
    }
    scanner.terminator()?;
    scanner.literal("DATA")?;
    scanner.terminator()?;
    let mut data = BTreeMap::new();
    while !scanner.try_literal("ENDSEC") {
        let id = scanner.reference()?;
        scanner.symbol(b'=')?;
        let instance = scanner.instance()?;
        scanner.terminator()?;
        if data.insert(id, instance).is_some() {
            return Err(invalid(format!("Part 21: duplicate entity #{id}")));
        }
    }
    scanner.terminator()?;
    scanner.literal("END-ISO-10303-21")?;
    scanner.terminator()?;
    scanner.trivia();
    if scanner.comment_unterminated {
        return Err(scanner.error("unterminated `/*` comment"));
    }
    if scanner.peek().is_some() {
        return Err(scanner.error("trailing content after `END-ISO-10303-21;`"));
    }
    // Part 21 forbids entity references in the header section.
    if header
        .iter()
        .flat_map(|record| &record.parameters)
        .any(has_reference)
    {
        return Err(invalid(
            "Part 21: entity reference in the HEADER section".to_string(),
        ));
    }
    for (id, instance) in &data {
        instance
            .records
            .iter()
            .flat_map(|record| &record.parameters)
            .try_for_each(|parameter| resolve(parameter, *id, &data))?;
    }
    Ok(Exchange { header, data })
}

fn has_reference(parameter: &Parameter) -> bool {
    match parameter {
        Parameter::Reference(_) => true,
        Parameter::List(items) => items.iter().any(has_reference),
        Parameter::Typed { parameter, .. } => has_reference(parameter),
        _ => false,
    }
}

fn resolve(parameter: &Parameter, id: u64, data: &BTreeMap<u64, Instance>) -> Result<()> {
    match parameter {
        Parameter::Reference(target) if !data.contains_key(target) => Err(invalid(format!(
            "Part 21: #{id} references undefined entity #{target}"
        ))),
        Parameter::List(items) => items.iter().try_for_each(|item| resolve(item, id, data)),
        Parameter::Typed { parameter, .. } => resolve(parameter, id, data),
        _ => Ok(()),
    }
}

fn is_keyword(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'!')
}

/// Cap on nested list / typed-parameter depth. A corrupt or crafted file with
/// pathological nesting errors here instead of overflowing the stack.
const MAX_DEPTH: usize = 256;

struct Scanner<'a> {
    bytes: &'a [u8],
    position: usize,
    depth: usize,
    /// Bytes of a stripped BOM, added back into reported offsets.
    base_offset: usize,
    /// A `/*` reached EOF with no `*/`; `trivia` stops advancing and `parse`
    /// turns it into an error rather than silently swallowing the tail.
    comment_unterminated: bool,
}

impl Scanner<'_> {
    fn error(&self, message: impl Into<String>) -> Error {
        invalid(format!(
            "Part 21: {} at byte {}",
            message.into(),
            self.base_offset + self.position
        ))
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn trivia(&mut self) {
        loop {
            while self.peek().is_some_and(|byte| byte.is_ascii_whitespace()) {
                self.position += 1;
            }
            if !self.comment_unterminated && self.bytes[self.position..].starts_with(b"/*") {
                match self.bytes[self.position + 2..]
                    .windows(2)
                    .position(|window| window == b"*/")
                {
                    Some(offset) => self.position += offset + 4,
                    None => {
                        self.comment_unterminated = true;
                        self.position = self.bytes.len();
                    }
                }
            } else {
                return;
            }
        }
    }

    fn symbol(&mut self, symbol: u8) -> Result<()> {
        self.trivia();
        if self.peek() == Some(symbol) {
            self.position += 1;
            Ok(())
        } else {
            Err(self.error(format!("expected `{}`", symbol as char)))
        }
    }

    fn terminator(&mut self) -> Result<()> {
        self.symbol(b';')
    }

    fn literal(&mut self, literal: &str) -> Result<()> {
        if self.try_literal(literal) {
            Ok(())
        } else {
            Err(self.error(format!("expected `{literal}`")))
        }
    }

    fn try_literal(&mut self, literal: &str) -> bool {
        self.trivia();
        let end = self.position + literal.len();
        if self.bytes[self.position..].starts_with(literal.as_bytes())
            && self.bytes.get(end).is_none_or(|byte| !is_keyword(*byte))
        {
            self.position = end;
            true
        } else {
            false
        }
    }

    fn keyword(&mut self) -> Result<String> {
        self.trivia();
        let start = self.position;
        if self.peek() == Some(b'!') {
            self.position += 1;
        }
        while self
            .peek()
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            self.position += 1;
        }
        if self.position == start || self.bytes[start].is_ascii_digit() {
            return Err(self.error("expected a keyword"));
        }
        Ok(from_utf8(&self.bytes[start..self.position])
            .unwrap()
            .to_owned())
    }

    fn reference(&mut self) -> Result<u64> {
        self.symbol(b'#')?;
        let start = self.position;
        while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
            self.position += 1;
        }
        if self.position == start {
            return Err(self.error("expected an entity id after `#`"));
        }
        if self.position > start + 1 && self.bytes[start] == b'0' {
            return Err(self.error("entity id has a leading zero"));
        }
        let id: u64 = from_utf8(&self.bytes[start..self.position])
            .unwrap()
            .parse()
            .map_err(|_| self.error("entity id out of range"))?;
        if id == 0 {
            return Err(self.error("entity id must be positive"));
        }
        Ok(id)
    }

    fn record(&mut self) -> Result<Record> {
        Ok(Record {
            keyword: self.keyword()?,
            parameters: self.parameters()?,
        })
    }

    fn instance(&mut self) -> Result<Instance> {
        self.trivia();
        if self.peek() != Some(b'(') {
            return Ok(Instance {
                records: vec![self.record()?],
            });
        }
        self.position += 1;
        let mut records = Vec::new();
        loop {
            self.trivia();
            match self.peek() {
                Some(b')') => {
                    self.position += 1;
                    break;
                }
                Some(_) => records.push(self.record()?),
                None => return Err(self.error("unterminated complex entity")),
            }
        }
        if records.is_empty() {
            return Err(self.error("empty complex entity"));
        }
        Ok(Instance { records })
    }

    fn parameters(&mut self) -> Result<Vec<Parameter>> {
        self.symbol(b'(')?;
        let mut parameters = Vec::new();
        self.trivia();
        if self.peek() == Some(b')') {
            self.position += 1;
            return Ok(parameters);
        }
        loop {
            parameters.push(self.parameter()?);
            self.trivia();
            match self.peek() {
                Some(b',') => self.position += 1,
                Some(b')') => {
                    self.position += 1;
                    return Ok(parameters);
                }
                _ => return Err(self.error("expected `,` or `)`")),
            }
        }
    }

    fn parameter(&mut self) -> Result<Parameter> {
        self.depth += 1;
        let result = if self.depth > MAX_DEPTH {
            Err(self.error("parameter nesting exceeds the depth limit"))
        } else {
            self.parameter_value()
        };
        self.depth -= 1;
        result
    }

    fn parameter_value(&mut self) -> Result<Parameter> {
        self.trivia();
        match self.peek() {
            None => Err(self.error("expected a parameter")),
            Some(b'$') => {
                self.position += 1;
                Ok(Parameter::Null)
            }
            Some(b'*') => {
                self.position += 1;
                Ok(Parameter::Derived)
            }
            Some(b'#') => Ok(Parameter::Reference(self.reference()?)),
            Some(b'\'') => Ok(Parameter::String(self.string()?)),
            Some(b'(') => Ok(Parameter::List(self.parameters()?)),
            Some(b'.') => Ok(Parameter::Enumeration(self.enumeration()?)),
            Some(byte) if byte == b'+' || byte == b'-' || byte.is_ascii_digit() => self.number(),
            Some(byte) if byte == b'!' || byte.is_ascii_alphabetic() => {
                let keyword = self.keyword()?;
                self.symbol(b'(')?;
                let parameter = Box::new(self.parameter()?);
                self.symbol(b')')?;
                Ok(Parameter::Typed { keyword, parameter })
            }
            Some(byte) => Err(self.error(format!("unexpected `{}`", byte as char))),
        }
    }

    fn string(&mut self) -> Result<String> {
        self.position += 1;
        let mut value = Vec::new();
        loop {
            match self.peek() {
                None => return Err(self.error("unterminated string")),
                Some(b'\'') if self.bytes.get(self.position + 1) == Some(&b'\'') => {
                    value.push(b'\'');
                    self.position += 2;
                }
                Some(b'\'') => {
                    self.position += 1;
                    return Ok(String::from_utf8_lossy(&value).into_owned());
                }
                // A raw control byte (a bare newline especially) is the mark of
                // a truncated quote; without this one runaway `'` swallows the
                // records that follow it as string content.
                Some(byte) if byte < 0x20 => {
                    return Err(self.error("control character in a string literal"));
                }
                Some(byte) => {
                    value.push(byte);
                    self.position += 1;
                }
            }
        }
    }

    fn enumeration(&mut self) -> Result<String> {
        self.position += 1;
        let start = self.position;
        while self
            .peek()
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            self.position += 1;
        }
        if self.position == start || self.peek() != Some(b'.') {
            return Err(self.error("malformed enumeration"));
        }
        if self.bytes[start].is_ascii_digit() {
            return Err(self.error("enumeration cannot start with a digit"));
        }
        let value = from_utf8(&self.bytes[start..self.position])
            .unwrap()
            .to_owned();
        self.position += 1;
        Ok(value)
    }

    fn number(&mut self) -> Result<Parameter> {
        let start = self.position;
        if matches!(self.peek(), Some(b'+') | Some(b'-')) {
            self.position += 1;
        }
        let mut real = false;
        let mut seen_digit = false;
        while let Some(byte) = self.peek() {
            match byte {
                b'0'..=b'9' => {
                    seen_digit = true;
                    self.position += 1;
                }
                b'.' => {
                    // Part 21 requires a digit before the decimal point, in the
                    // signed form (`-.5`) as much as the bare one (`.5`, which
                    // never reaches here — `parameter` sends it to `enumeration`).
                    if !seen_digit {
                        return Err(self.error("real literal needs a digit before `.`"));
                    }
                    real = true;
                    self.position += 1;
                }
                b'e' | b'E' => {
                    real = true;
                    self.position += 1;
                    if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                        self.position += 1;
                    }
                }
                _ => break,
            }
        }
        let text = from_utf8(&self.bytes[start..self.position]).unwrap();
        if real {
            match text.parse::<f64>() {
                // `str::parse` maps an overflowing literal to ±inf and an
                // underflowing one to 0.0, neither an error; reject both so a
                // bad literal never reaches geometry as an infinite or a
                // silently-zeroed coordinate.
                Ok(value)
                    if value.is_finite()
                        && (value != 0.0
                            || !text.bytes().any(|b| b.is_ascii_digit() && b != b'0')) =>
                {
                    Ok(Parameter::Real(value))
                }
                _ => Err(self.error(format!("malformed or out-of-range real `{text}`"))),
            }
        } else {
            text.parse()
                .map(Parameter::Integer)
                .map_err(|_| self.error(format!("malformed integer `{text}`")))
        }
    }
}
