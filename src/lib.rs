use pyo3::prelude::*;

pub mod arch;
pub mod loader;
pub mod objective;
pub mod predicates;

#[allow(clippy::useless_conversion)]
mod python;
pub mod tree;

pub use loader::ModelLoader;
pub use objective::Objective;
pub use predicates::{Condition, Predicate};
pub use tree::{FeatureTreeBuilder, GradientBoostedDecisionTrees, VecTreeNodes};

#[pymodule]
fn quickgrove(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let internal = py.import_bound("quickgrove._internal")?;
    m.add("_internal", internal)?;
    Ok(())
}

#[pymodule]
fn _internal(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(python::json_load))?;
    m.add_class::<python::PyGradientBoostedDecisionTrees>()?;
    m.add_class::<python::Feature>()?;
    Ok(())
}
