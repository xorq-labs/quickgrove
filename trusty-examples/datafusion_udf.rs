use arrow::array::{ArrayRef, Float32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use datafusion_common::Result as DataFusionResult;
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use serde_json::Value;
use std::any::Any;
use std::error::Error;
use std::sync::Arc;
use trusty::loader::ModelLoader;
use trusty::tree::FeatureType;
use trusty::GradientBoostedDecisionTrees;

const MODEL_JSON: &str = r#"{
    "learner": {
        "feature_names": ["feature0", "feature1"],
        "feature_types": ["float", "float"],
        "objective": {"name": "reg:squarederror"},
        "learner_model_param": {
            "base_score": "0.5",
            "objective": {"name": "reg:squarederror"}
        },
        "gradient_booster": {
            "model": {
                "trees": [
                    {
                        "split_indices": [0, -1, -1],
                        "split_conditions": [0.5, 0.0, 0.0],
                        "left_children": [1, 4294967295, 4294967295],
                        "right_children": [2, 4294967295, 4294967295],
                        "base_weights": [0.0, -1.0, 1.0],
                        "default_left": [0, 0, 0],
                        "sum_hessian": [0, 0, 0]
                    }
                ]
            }
        }
    }
}"#;

#[derive(Debug, Clone)]
struct TrustyUDF {
    signature: Signature,
    trees: Arc<GradientBoostedDecisionTrees>,
    return_type: DataType,
}

impl PartialEq for TrustyUDF {
    fn eq(&self, other: &Self) -> bool {
        self.signature == other.signature
            && self.return_type == other.return_type
            && Arc::ptr_eq(&self.trees, &other.trees)
    }
}

impl Eq for TrustyUDF {}

impl std::hash::Hash for TrustyUDF {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.signature.hash(state);
        self.return_type.hash(state);
        Arc::as_ptr(&self.trees).hash(state);
    }
}

impl TrustyUDF {
    fn new() -> Result<Self, Box<dyn Error>> {
        let model_data: Value = serde_json::from_str(MODEL_JSON)?;
        let model: GradientBoostedDecisionTrees =
            GradientBoostedDecisionTrees::json_loads(&model_data)?;
        let mut arg_types = Vec::new();
        for feature_types in model.feature_types.iter() {
            match feature_types {
                FeatureType::Float => arg_types.push(DataType::Float32),
                FeatureType::Int => arg_types.push(DataType::Int64),
                FeatureType::Indicator => arg_types.push(DataType::Boolean),
            }
        }
        Ok(Self {
            signature: Signature::exact(arg_types, Volatility::Immutable),
            trees: Arc::new(model),
            return_type: DataType::Float32,
        })
    }
}

impl ScalarUDFImpl for TrustyUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "predict"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DataFusionResult<DataType> {
        Ok(self.return_type.clone())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DataFusionResult<ColumnarValue> {
        let arrays: Vec<ArrayRef> = args
            .args
            .iter()
            .map(|arg| match arg {
                ColumnarValue::Array(arr) => Ok(Arc::clone(arr)),
                ColumnarValue::Scalar(s) => s.to_array(),
            })
            .collect::<DataFusionResult<Vec<_>>>()?;

        let predictions = self.trees.predict_arrays(&arrays)?;
        Ok(ColumnarValue::Array(Arc::new(predictions)))
    }
}

fn create_trusty_udf() -> Result<ScalarUDF, Box<dyn Error>> {
    let udf = TrustyUDF::new()?;
    Ok(ScalarUDF::from(udf))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let ctx = SessionContext::new();

    let schema = Arc::new(Schema::new(vec![
        Field::new("feature0", DataType::Float32, false),
        Field::new("feature1", DataType::Float32, false),
    ]));

    let feature0 = Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]));
    let feature1 = Arc::new(Float32Array::from(vec![0.5, 1.5, 2.5, 3.5]));
    let batch = RecordBatch::try_new(schema.clone(), vec![feature0, feature1])?;

    ctx.register_batch("test_table", batch)?;
    let predict_udf = create_trusty_udf()?;
    ctx.register_udf(predict_udf);

    let df = ctx
        .sql(
            "SELECT *, predict(feature0, feature1) as prediction 
             FROM test_table",
        )
        .await?;

    df.show().await?;
    Ok(())
}
