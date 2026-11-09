//! Errors shared by the planner, DataFusion integration, and Python boundary.

use std::fmt;

/// A malformed query, unsupported region, or plan the configuration cannot run.
#[derive(Debug, Clone, PartialEq)]
pub struct CompileError {
    message: String,
    refusal: Option<Box<Refusal>>,
    kind: ErrorKind,
}

/// What a [`CompileError`] refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// A query Quail cannot answer as written.
    Invalid,
    /// A region shape DataFusion runs around a smaller model region instead.
    Unsupported,
}

impl CompileError {
    /// An error with the message and no refusal.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            refusal: None,
            kind: ErrorKind::Invalid,
        }
    }

    /// A region shape DataFusion must run, with a concrete reason.
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            refusal: None,
            kind: ErrorKind::Unsupported,
        }
    }

    /// An error carrying a refusal; its message is the first reason, or the
    /// constraint when no reasons were supplied.
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

    pub fn refusal(&self) -> Option<&Refusal> {
        self.refusal.as_deref()
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Whether the error refuses a region shape DataFusion can run.
    pub fn is_unsupported(&self) -> bool {
        self.kind == ErrorKind::Unsupported
    }

    /// Append diagnostic context without changing the error's kind or refusal.
    pub fn with_context(mut self, context: impl fmt::Display) -> Self {
        self.message = format!("{}:\n{context}", self.message);
        self
    }
}

/// A plan the configuration cannot run within its limits.
#[derive(Debug, Clone, PartialEq)]
pub struct Refusal {
    pub reasons: Vec<String>,
    pub constraint: String,
    pub needed: f64,
    pub available: f64,
    pub unit: String,
}

impl Refusal {
    /// A refusal with one reason.
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

/// Return a `CompileError` with the message.
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
        assert_eq!(error.refusal(), Some(&refusal));
        assert_eq!(error.kind(), ErrorKind::Invalid);
    }

    #[test]
    fn context_preserves_kind_and_message_format() {
        let error = CompileError::unsupported("These operators must run outside the model region")
            .with_context("plan");
        assert_eq!(
            error.to_string(),
            "These operators must run outside the model region:\nplan"
        );
        assert!(error.is_unsupported());
        assert_eq!(error.refusal(), None);
        assert_eq!(
            CompileError::refused(Refusal::new("reason", "memory", 2.0, 1.0, "GB")).message(),
            "reason"
        );
    }
}
