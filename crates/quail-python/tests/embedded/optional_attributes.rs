//! Optional engine properties must not hide Python initialization failures.

use pyo3::prelude::*;
use pyo3::types::PyModule;

fn engine(attribute: &str, behavior: &str) -> Py<PyAny> {
    crate::start_python();
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            c"
class Engine:
    turn = ('', '')
    layout = ''
    rules = None

def make(attribute, behavior):
    class Custom(Engine):
        pass
    if behavior == 'none':
        setattr(Custom, attribute, None)
    elif behavior != 'missing':
        def getter(self):
            if behavior == 'attribute_error':
                raise AttributeError(attribute)
            if behavior == 'os_error':
                raise OSError('sentinel: ' + attribute)
            if behavior in ('cause', 'context'):
                try:
                    raise OSError('source cause: ' + attribute)
                except OSError as error:
                    if behavior == 'cause':
                        raise RuntimeError('sentinel: ' + attribute) from error
                    raise RuntimeError('sentinel: ' + attribute)
            raise RuntimeError('sentinel: ' + attribute)
        setattr(Custom, attribute, property(getter))
    return Custom()
",
            c"optional_engine.py",
            c"optional_engine",
        )
        .expect("test module");
        module
            .getattr("make")
            .unwrap()
            .call1((attribute, behavior))
            .unwrap()
            .unbind()
    })
}

#[test]
fn optional_property_failures_preserve_the_exception_and_traceback() {
    for attribute in ["tokenizer_file", "settings"] {
        for behavior in ["runtime_error", "os_error", "cause", "context"] {
            let error =
                quail_python::executor_with_engine(engine(attribute, behavior)).unwrap_err();
            let message = error.to_string();
            assert!(
                message.contains(&format!("sentinel: {attribute}")),
                "{message}"
            );
            assert!(message.contains("optional_engine.py"), "{message}");
            assert!(message.contains("in getter"), "{message}");
            if matches!(behavior, "cause" | "context") {
                assert!(
                    message.contains(&format!("source cause: {attribute}")),
                    "{message}"
                );
                let chain = if behavior == "cause" {
                    "direct cause"
                } else {
                    "During handling"
                };
                assert!(message.contains(chain), "{message}");
            }
        }
    }
}

#[test]
fn missing_and_none_optional_properties_retain_fallback() {
    for attribute in ["tokenizer_file", "settings"] {
        for behavior in ["missing", "none", "attribute_error"] {
            quail_python::executor_with_engine(engine(attribute, behavior))
                .expect("fallback executor");
        }
    }
}
