use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct CompileError {
    message: String,
    refusal: Option<Box<Refusal>>,
    kind: ErrorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Invalid,
    Unsupported,
}

impl CompileError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            refusal: None,
            kind: ErrorKind::Invalid,
        }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            refusal: None,
            kind: ErrorKind::Unsupported,
        }
    }

    pub fn refused(refusal: Refusal) -> Self {
        Self {
            message: refusal
                .reasons
                .first()
                .unwrap_or(&refusal.constraint)
                .clone(),
            refusal: Some(Box::new(refusal)),
            kind: ErrorKind::Invalid,
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn into_refusal(self) -> Result<Refusal, CompileError> {
        match self.refusal {
            Some(refusal) => Ok(*refusal),
            None => Err(self),
        }
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn is_unsupported(&self) -> bool {
        self.kind == ErrorKind::Unsupported
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Refusal {
    pub reasons: Vec<String>,
    pub constraint: String,
    pub needed: f64,
    pub available: f64,
    pub unit: String,
}

impl Refusal {
    pub fn new(
        reason: impl Into<String>,
        constraint: &str,
        needed: f64,
        available: f64,
        unit: &str,
    ) -> Self {
        Self {
            reasons: vec![reason.into()],
            constraint: constraint.to_string(),
            needed,
            available,
            unit: unit.to_string(),
        }
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for CompileError {}

pub fn compile_error<T>(message: impl Into<String>) -> Result<T, CompileError> {
    Err(CompileError::new(message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusal_without_reasons_uses_constraint() {
        let mut refusal = Refusal::new("reason", "memory", 2.0, 1.0, "GB");
        refusal.reasons.clear();
        let error = CompileError::refused(refusal.clone());
        assert_eq!(error.message(), "memory");
        assert_eq!(error.kind(), ErrorKind::Invalid);
        assert_eq!(error.into_refusal(), Ok(refusal));
    }

    #[test]
    fn a_refusal_without_reasons_reports_its_constraint() {
        assert_eq!(
            CompileError::refused(Refusal::new("reason", "memory", 2.0, 1.0, "GB")).message(),
            "reason"
        );
    }
}
