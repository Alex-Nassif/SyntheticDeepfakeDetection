
from collections import Counter

import numpy as np
from datasets import load_dataset

from data_processor import get_input, processing_input
from model_setup import configure_model, run_model
from evaluation import generate_conclusion, calculate_metrics


def main():
    SAMPLE_COUNT = 500
    THRESHOLD = 0.47

    print("Loading CIFAKE dataset...")

    dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset")
    test_data = dataset["test"].shuffle(seed=42)

    from collections import Counter

    print("Full test-set labels:", Counter(test_data["label"]))
    print("First 20 shuffled labels:", test_data["label"][:20])
    sample_count = min(SAMPLE_COUNT, len(test_data))

    label_feature = test_data.features["label"]
    label_names = label_feature.names

    print("Label feature:", label_feature)
    print("Label names:", label_names)

    if "FAKE" not in label_names or "REAL" not in label_names:
        raise ValueError(f"Unexpected dataset labels: {label_names}")

    print("\nLoading GenD model...")
    model = configure_model()

    predictions = []
    true_labels = []

    print(f"\nEvaluating {sample_count} images...\n")

    for i in range(sample_count):
        sample = test_data[i]

        # Convert image to RGB.
        image = sample["image"].convert("RGB")

        # Dataset labels: 0 = FAKE, 1 = REAL.
        raw_label = int(sample["label"])
        label_name = label_feature.int2str(raw_label)

        # Evaluation labels: 0 = REAL, 1 = FAKE.
        actual_label = 1 if label_name == "FAKE" else 0

        # Convert image from HWC to CHW.
        input_array = np.array(image).transpose(2, 0, 1)

        # Prepare input and run inference.
        input_data = get_input(input_array)

        processed_input = processing_input(
            input_data,
            model,
            pixel_range="0_255",
            high_pass=False,
        )

        result = run_model(model, processed_input)
        deepfake_score = result["deepfake_score"]


        conclusion = generate_conclusion(
            deepfake_score,
            THRESHOLD,
        )

        prediction = conclusion["prediction"]
        score = conclusion["deepfake_score"]

        predictions.append(int(prediction))
        true_labels.append(actual_label)

        if i < 5:
            print(
                f"Sample {i}: "
                f"raw_label={raw_label}, "
                f"label_name={label_name}, "
                f"actual={actual_label}, "
                f"score={score:.4f}, "
                f"prediction={prediction}, "
                f"inference_time={result['inference_seconds']:.4f}s"
            )

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{sample_count}")

    print("\nPrediction counts:", Counter(predictions))
    print("Ground-truth counts:", Counter(true_labels))

    if len(set(true_labels)) < 2:
        raise ValueError(
            "Only one ground-truth class was found. "
            "Check the dataset labels and test split."
        )
    metrics = calculate_metrics(predictions, true_labels)

    print("\nCIFAKE Test Results")
    print(f"Sample count: {sample_count}")
    print(f"Threshold: {THRESHOLD}")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")


if __name__ == "__main__":
    main()
