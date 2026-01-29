#!/usr/bin/env python3
"""
PaddleDetection to ONNX Converter

Converts trained PaddleDetection models to ONNX format for deployment
with ONNX Runtime, TensorRT, or DeepStream.

Usage:
    # Export inference model first (from train.py)
    python export_onnx.py --input ./output/inference_model/ppyoloe_l

    # Direct export from weights
    python export_onnx.py --model ppyoloe_l --weights ./output/ppyoloe_l/best_model.pdparams

    # Specify input size
    python export_onnx.py --input ./output/inference_model/ppyoloe_l --input-size 640 640

    # Validate ONNX model
    python export_onnx.py --validate --onnx ./output/ppyoloe_l.onnx
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime


def check_dependencies():
    """Check if required packages are installed"""
    missing = []

    try:
        import paddle2onnx
    except ImportError:
        missing.append("paddle2onnx")

    try:
        import onnx
    except ImportError:
        missing.append("onnx")

    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime-gpu")

    if missing:
        print("Missing dependencies. Install with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def export_paddle_to_inference(paddle_dir: Path, config_path: Path,
                               weights_path: Path, output_dir: Path,
                               input_size: list) -> bool:
    """Export Paddle model to inference format"""
    os.chdir(paddle_dir)

    cmd = [
        sys.executable, "tools/export_model.py",
        "-c", str(config_path),
        "-o", f"weights={weights_path}",
        f"TestReader.inputs_def.image_shape=[3,{input_size[0]},{input_size[1]}]",
        "--output_dir", str(output_dir)
    ]

    print(f"\nExporting Paddle model to inference format...")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Export failed: {e}")
        return False


def convert_to_onnx(model_dir: Path, output_path: Path, opset_version: int = 11) -> bool:
    """Convert Paddle inference model to ONNX"""
    model_file = model_dir / "model.pdmodel"
    params_file = model_dir / "model.pdiparams"

    if not model_file.exists():
        print(f"Error: model.pdmodel not found in {model_dir}")
        return False

    if not params_file.exists():
        print(f"Error: model.pdiparams not found in {model_dir}")
        return False

    cmd = [
        "paddle2onnx",
        "--model_dir", str(model_dir),
        "--model_filename", "model.pdmodel",
        "--params_filename", "model.pdiparams",
        "--opset_version", str(opset_version),
        "--save_file", str(output_path),
        "--enable_onnx_checker", "True"
    ]

    print(f"\nConverting to ONNX...")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\nONNX model saved: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
        return False
    except FileNotFoundError:
        print("Error: paddle2onnx not found. Install with:")
        print("  pip install paddle2onnx")
        return False


def validate_onnx(onnx_path: Path) -> bool:
    """Validate ONNX model"""
    try:
        import onnx
        from onnx import checker

        print(f"\nValidating ONNX model: {onnx_path}")

        model = onnx.load(str(onnx_path))
        checker.check_model(model)

        # Print model info
        print(f"\n  Model is valid")
        print(f"  Opset version: {model.opset_import[0].version}")

        print(f"\n  Inputs:")
        for inp in model.graph.input:
            shape = [d.dim_value if d.dim_value > 0 else "dynamic"
                     for d in inp.type.tensor_type.shape.dim]
            print(f"    {inp.name}: {shape}")

        print(f"\n  Outputs:")
        for out in model.graph.output:
            shape = [d.dim_value if d.dim_value > 0 else "dynamic"
                     for d in out.type.tensor_type.shape.dim]
            print(f"    {out.name}: {shape}")

        # Count operations
        ops = {}
        for node in model.graph.node:
            ops[node.op_type] = ops.get(node.op_type, 0) + 1

        print(f"\n  Operations: {len(ops)} types, {sum(ops.values())} total")

        # Check for TensorRT compatibility issues
        problematic_ops = ["NonMaxSuppression", "Loop", "If", "Scan"]
        found_problematic = [op for op in problematic_ops if op in ops]
        if found_problematic:
            print(f"\n  Warning: Found ops that may need plugins for TensorRT: {found_problematic}")

        return True

    except Exception as e:
        print(f"Validation failed: {e}")
        return False


def test_inference(onnx_path: Path, image_path: Path = None) -> bool:
    """Test ONNX model inference with ONNX Runtime"""
    try:
        import onnxruntime as ort
        import numpy as np

        print(f"\nTesting inference with ONNX Runtime...")

        # Create session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(str(onnx_path), providers=providers)

        print(f"  Active provider: {session.get_providers()[0]}")

        # Get input info
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape

        # Replace dynamic dimensions with test values
        test_shape = []
        for dim in input_shape:
            if isinstance(dim, int) and dim > 0:
                test_shape.append(dim)
            else:
                test_shape.append(1 if len(test_shape) == 0 else 640)

        print(f"  Input: {input_name}, shape: {test_shape}")

        # Create dummy input
        dummy_input = np.random.randn(*test_shape).astype(np.float32)

        # Run inference
        import time
        start = time.time()
        outputs = session.run(None, {input_name: dummy_input})
        elapsed = (time.time() - start) * 1000

        print(f"\n  Inference successful!")
        print(f"  Time: {elapsed:.2f} ms")
        print(f"  Outputs:")
        for i, out in enumerate(outputs):
            print(f"    [{i}]: shape={out.shape}, dtype={out.dtype}")

        return True

    except Exception as e:
        print(f"Inference test failed: {e}")
        return False


def simplify_onnx(input_path: Path, output_path: Path) -> bool:
    """Simplify ONNX model using onnxsim"""
    try:
        from onnxsim import simplify
        import onnx

        print(f"\nSimplifying ONNX model...")

        model = onnx.load(str(input_path))
        model_simplified, check = simplify(model)

        if check:
            onnx.save(model_simplified, str(output_path))
            print(f"Simplified model saved: {output_path}")

            # Compare sizes
            orig_size = input_path.stat().st_size / (1024 * 1024)
            new_size = output_path.stat().st_size / (1024 * 1024)
            print(f"  Original: {orig_size:.2f} MB")
            print(f"  Simplified: {new_size:.2f} MB")

            return True
        else:
            print("Simplification check failed, model may be incorrect")
            return False

    except ImportError:
        print("onnxsim not installed. Install with: pip install onnxsim")
        return False
    except Exception as e:
        print(f"Simplification failed: {e}")
        return False


def copy_to_deployment(onnx_path: Path, target_dir: Path, model_name: str):
    """Copy ONNX model to deployment directory"""
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy ONNX file
    target_path = target_dir / f"{model_name}.onnx"
    shutil.copy2(onnx_path, target_path)
    print(f"\nCopied to: {target_path}")

    # Create info file
    info_file = target_dir / f"{model_name}_info.txt"
    with open(info_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Source: {onnx_path}\n")
        f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Size: {onnx_path.stat().st_size / (1024*1024):.2f} MB\n")

    return target_path


def main():
    parser = argparse.ArgumentParser(description="Convert PaddleDetection models to ONNX")
    parser.add_argument("--input", "-i", type=str,
                        help="Path to Paddle inference model directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output ONNX file path")
    parser.add_argument("--model", "-m", type=str,
                        help="Model name (for direct export from weights)")
    parser.add_argument("--weights", "-w", type=str,
                        help="Path to model weights (.pdparams)")
    parser.add_argument("--paddle-dir", type=str, default="./PaddleDetection",
                        help="PaddleDetection directory")
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 640],
                        help="Input size (height width)")
    parser.add_argument("--opset", type=int, default=11,
                        help="ONNX opset version")
    parser.add_argument("--validate", "-v", action="store_true",
                        help="Validate ONNX model")
    parser.add_argument("--onnx", type=str,
                        help="Path to ONNX file for validation/testing")
    parser.add_argument("--test", "-t", action="store_true",
                        help="Test inference with ONNX Runtime")
    parser.add_argument("--simplify", "-s", action="store_true",
                        help="Simplify ONNX model with onnxsim")
    parser.add_argument("--deploy-dir", type=str, default=None,
                        help="Copy to deployment directory (e.g., ../inference/weights)")

    args = parser.parse_args()

    # Validate only mode
    if args.validate and args.onnx:
        onnx_path = Path(args.onnx)
        validate_onnx(onnx_path)
        if args.test:
            test_inference(onnx_path)
        return

    # Check dependencies
    if not check_dependencies():
        return

    # Determine input path
    if args.input:
        model_dir = Path(args.input).resolve()
        if not model_dir.exists():
            print(f"Error: Model directory not found: {model_dir}")
            return
    elif args.model and args.weights:
        # TODO: First export to inference format
        print("Direct export from weights not implemented yet.")
        print("First export using train.py --export, then run this script.")
        return
    else:
        print("Error: Either --input or (--model and --weights) required")
        return

    # Determine output path
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = model_dir.parent / f"{model_dir.name}.onnx"

    # Convert to ONNX
    success = convert_to_onnx(model_dir, output_path, args.opset)

    if not success:
        return

    # Validate
    if args.validate or True:  # Always validate
        validate_onnx(output_path)

    # Simplify
    if args.simplify:
        simplified_path = output_path.with_suffix(".simplified.onnx")
        if simplify_onnx(output_path, simplified_path):
            output_path = simplified_path

    # Test inference
    if args.test:
        test_inference(output_path)

    # Copy to deployment directory
    if args.deploy_dir:
        deploy_dir = Path(args.deploy_dir).resolve()
        model_name = model_dir.name
        copy_to_deployment(output_path, deploy_dir, model_name)

    print(f"\n{'='*60}")
    print(f"ONNX export complete!")
    print(f"Output: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
