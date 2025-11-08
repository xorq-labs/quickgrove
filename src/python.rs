use crate::loader::ModelLoader;
use crate::tree::{GradientBoostedDecisionTrees, PredictorConfig, VecTreeNodes};
use crate::Condition;
use crate::Predicate;
use arrow::array::Array;
use arrow::array::ArrayRef;
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::pyarrow::PyArrowType;
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::types::PyType;
use pyo3_arrow::error::PyArrowResult;
use pyo3_arrow::PyArray;
use std::path::PathBuf;
use std::sync::Arc;

#[pyclass]
#[derive(Clone)]
pub struct Feature {
    name: String,
}

#[pymethods]
impl Feature {
    #[new]
    fn new(name: &str) -> Self {
        Feature {
            name: name.to_string(),
        }
    }

    fn __lt__(&self, other: f64) -> (String, bool, f64) {
        (self.name.clone(), false, other) // false means LessThan
    }

    fn __ge__(&self, other: f64) -> (String, bool, f64) {
        (self.name.clone(), true, other) // true means GreaterThanOrEqual
    }
}

#[pyclass]
pub struct PyFeatureTree {
    tree: VecTreeNodes,
}

#[pymethods]
impl PyFeatureTree {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.tree))
    }
}

#[pyclass]
pub struct PyGradientBoostedDecisionTrees {
    model: Arc<GradientBoostedDecisionTrees>,
}

#[pymethods]
impl PyGradientBoostedDecisionTrees {
    #[new]
    fn new(model_json: &str) -> PyResult<Self> {
        let model_data: serde_json::Value = serde_json::from_str(model_json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let model = GradientBoostedDecisionTrees::json_loads(&model_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyGradientBoostedDecisionTrees {
            model: Arc::new(model),
        })
    }

    #[classmethod]
    fn json_load(_cls: Py<PyType>, path: PathBuf) -> PyResult<Self> {
        let str_path = path
            .to_str()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid path"))?;
        let model = GradientBoostedDecisionTrees::json_load(str_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyGradientBoostedDecisionTrees {
            model: Arc::new(model),
        })
    }

    #[pyo3(signature = (py_record_batches, *, row_chunk_size=64, tree_chunk_size=8))]
    fn predict_batches(
        &self,
        py: Python,
        py_record_batches: &Bound<'_, PyList>,
        row_chunk_size: usize,
        tree_chunk_size: usize,
    ) -> PyArrowResult<PyObject> {
        let mut batches = Vec::with_capacity(py_record_batches.len());
        // Need this clone to make config work. perhaps, another way to avoid it?
        let model = Arc::new({
            let mut m = (*self.model).clone();
            m.set_config(PredictorConfig {
                row_chunk_size,
                tree_chunk_size,
            });
            m
        });

        for py_batch in py_record_batches.iter() {
            let py_arrow_type = py_batch.extract::<PyArrowType<RecordBatch>>()?;
            let record_batch = py_arrow_type.0;
            let arrays: Vec<ArrayRef> = record_batch
                .columns()
                .iter()
                .map(|col| {
                    if col.data_type() == &DataType::Float64 {
                        cast(col, &DataType::Float32).unwrap()
                    } else {
                        Arc::clone(col)
                    }
                })
                .collect();
            let new_schema = Schema::new(
                record_batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| {
                        if field.data_type() == &DataType::Float64 {
                            Arc::new(Field::new(
                                field.name(),
                                DataType::Float32,
                                field.is_nullable(),
                            ))
                        } else {
                            field.clone()
                        }
                    })
                    .collect::<Vec<Arc<Field>>>(),
            );
            let float32_batch = RecordBatch::try_new(Arc::new(new_schema), arrays).unwrap();
            batches.push(float32_batch);
        }

        let predictions_array = model
            .predict_batches(&batches)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let field = Field::new("predictions", DataType::Float32, false);
        Ok(PyArray::new(Arc::new(predictions_array), Arc::new(field)).to_pyarrow(py)?)
    }

    fn prune(&self, predicates: &Bound<'_, PyList>) -> PyResult<Self> {
        let mut predicate = Predicate::new();
        for pred in predicates.iter() {
            let (feature_name, is_gte, threshold): (String, bool, f64) = pred.extract()?;
            let condition = if is_gte {
                Condition::GreaterThanOrEqual(threshold)
            } else {
                Condition::LessThan(threshold)
            };
            predicate.add_condition(feature_name, condition);
        }
        Ok(Self {
            model: Arc::new((*self.model).prune(&predicate)),
        })
    }
    #[getter]
    fn feature_names(&self) -> Vec<String> {
        (*self.model.feature_names).clone()
    }

    /// Get the types of features used by the model
    #[getter]
    fn feature_types(&self) -> Vec<String> {
        self.model
            .feature_types
            .iter()
            .map(|ft| ft.to_string())
            .collect()
    }

    /// Get the indices of required features
    #[getter]
    fn required_features(&self) -> Vec<usize> {
        self.model.required_features.iter().copied().collect()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.model))
    }

    fn predict_arrays(&self, py: Python, py_arrays: &Bound<'_, PyList>) -> PyArrowResult<PyObject> {
        let mut arrays = Vec::with_capacity(py_arrays.len());

        for py_array in py_arrays.iter() {
            let arrow_array: PyArray = py_array.extract()?;
            let array_ref = arrow_array.array();

            let processed_array = if array_ref.data_type() == &DataType::Float64 {
                Arc::new(cast(&array_ref, &DataType::Float32).unwrap())
            } else {
                array_ref.clone()
            };

            arrays.push(processed_array);
        }

        let predictions_array = self
            .model
            .predict_arrays(&arrays)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let predictions_ref: ArrayRef = Arc::new(predictions_array);
        let field = Arc::new(Field::new("predictions", DataType::Float32, false));

        Ok(PyArray::new(predictions_ref, field).to_pyarrow(py)?)
    }

    #[pyo3(signature = (tree_index=None))]
    fn tree_info(&self, tree_index: Option<usize>) -> PyResult<PyFeatureTree> {
        match tree_index {
            Some(idx) if idx < self.model.trees.len() => Ok(PyFeatureTree {
                tree: self.model.trees[idx].clone(),
            }),
            Some(idx) => Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Tree index {} out of range",
                idx
            ))),
            None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "tree_index is required",
            )),
        }
    }

    /// Returns the model as an XGBoost-compatible JSON string
    fn to_json(&self) -> PyResult<String> {
        self.model
            .to_json_string()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}

#[pyfunction]
pub fn json_load(path: PathBuf) -> PyResult<PyGradientBoostedDecisionTrees> {
    let str_path = path
        .to_str()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid path"))?;
    let model = GradientBoostedDecisionTrees::json_load(str_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(PyGradientBoostedDecisionTrees {
        model: Arc::new(model),
    })
}
