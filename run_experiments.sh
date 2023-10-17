#!/usr/bin/env bash

transformations=("no_transformation" "tf_1" "tf_2" "tf_3" "tf_4" "tf_5" "tf_6" "tf_7" "tf_8" "tf_9" "tf_10" "tf_11")
datasets=("CodeXGLUE" "VulDeePecker")
techniques=("UniXcoder", "CoTexT" "VulBERTa" "PLBart", "CodeBERT", "GraphCodeBERT")

# Algorithms 1 and 2
for dataset in "${datasets[@]}"
do
    for technique in "${techniques[@]}"
    do
        for transformation in "${transformations[@]}"
        do
            ./scripts/$dataset/$technique/run.sh "$transformation"
        done
    done
done