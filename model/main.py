import numpy as np
from PIL import Image

from data_processor import get_input, processing_input
from model_setup import configure_model, run_model
from evaluation import generate_conclusion, calculate_metrics


def main():

    model = configure_model()



    image_path = "images/test_image.jpg"

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        input_array = np.array(image)

    # Convert (height, width, channels) to (channels, height, width).
    input_array = np.transpose(input_array, (2, 0, 1))

    input_arrays = [input_array]

    # No ground-truth labels needed for inference.
    true_labels = None


    #Replace with corresponding labels .
    #Real = 0, deepfake = 1.
    true_labels = None

    predictions = []

    for index, input_array in enumerate(input_arrays):
        #validate the Channel, Height, Width array and collect metadata.
        input_data = get_input(input_array)

        #apply GenD preprocessing and add a batch dimension.
        processed_input = processing_input(
            input_data,
            model,
            pixel_range="0_255",
        )

        #run the detector and measure inference time.
        result = run_model(model, processed_input)

        #convert the score into a predicted label.
        conclusion = generate_conclusion(
            result["deepfake_score"],
            threshold=0.5,
        )

        predictions.append(conclusion["prediction"])

        print(f"\nSample {index + 1}")
        print(f"Conclusion: {conclusion['conclusion']}")
        print(f"Deepfake score: {result['deepfake_score']:.4f}")
        print(f"Inference time: {result['inference_seconds']:.4f} seconds")

    #evaluate performance only when known labels are supplied.
    if true_labels is not None:
        metrics = calculate_metrics(predictions, true_labels)

        print("\nDataset metrics")
        for name, value in metrics.items():
            print(f"{name}: {value}")
    else:
        metrics = None
        print("\nPerformance metrics unavailable: no true labels supplied.")

    return {
        "predictions": predictions,
        "metrics": metrics,
    }



if __name__ == "__main__":
    main()