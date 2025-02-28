import argparse
import json
from .polygon_saver import PolygonSaver
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO results to JSON files.')
    parser.add_argument('--model', default='yolov8n-seg.pt', help='Path to YOLO model weights file (default is yolov8n)')
    parser.add_argument('--images', required=True, help='Path to folder containing images')
    parser.add_argument('--config', required=False, help='Path to json file containing yolo prediction configuration')
    args = parser.parse_args()

    # Load YOLO arguments from config file
    yolo_arguments = json.load(open(args.config, 'r')) if args.config else {}

    # Use the input images directory as the output directory for JSON files
    output_dir = args.images

    # Instantiate the PolygonSaver class
    polygon_saver = PolygonSaver()

    # Load the YOLO model
    yolo_model = YOLO(args.model)

    # Get results from YOLO model predictions
    results = yolo_model.predict(args.images, save=True, **yolo_arguments)

    # Generate JSON files with results
    polygon_saver.generate_json_with_results(results, output_dir)

if __name__ == "__main__":
    main()
