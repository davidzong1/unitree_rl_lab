import os
import time
import argparse
from casadi import *
from cusadi.src import *
import subprocess
from pathlib import Path
import shutil

# TODO: REPEAT BENCHMARK WITH TORCH VMAP INSTEAD OF VECTORIZING WITH DICT.


def main(args):
    casadi_fns = []
    casadi_fns_file_name: list[str] = []
    if args.func_dir:
        fn_dir = args.func_dir
    else:
        fn_dir = CUSADI_BENCHMARK_DIR if args.codegen_benchmark_fns else CUSADI_FUNCTION_DIR
    if args.clean_compile:
        print("Cleaning codegen directory...")
        if os.path.isdir(CUSADI_CODEGEN_DIR):
            shutil.rmtree(CUSADI_CODEGEN_DIR)
        build_root = Path(CUSADI_ROOT_DIR)
        build_dir = build_root / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
    os.makedirs(CUSADI_CODEGEN_DIR, exist_ok=True)
    for filename in os.listdir(fn_dir):
        f = os.path.join(fn_dir, filename)
        if os.path.isfile(f) and f.endswith(".casadi"):
            if args.fn_name == "all" or args.fn_name in f:
                filename_without_ext = os.path.splitext(os.path.basename(f))[0]
                casadi_fns_file_name.append(filename_without_ext)
                print("CasADi function found: ", f)
                casadi_fns.append(casadi.Function.load(f))
    cnt: int = 0
    for f in casadi_fns:
        cuda_name = casadi_fns_file_name[cnt]
        if args.precision:
            print("Generating double code")
            generateCUDACodeDouble(f, cuda_name=cuda_name)
        else:
            print("Generating float code")
            generateCUDACodeFloat(f, cuda_name=cuda_name)
        # generateCUDACodeFloat(f)
        # generateCUDACodeDouble(f)
        if args.gen_pytorch:
            generatePytorchCode(f, cuda_name=cuda_name)
        cnt += 1

    generateCMakeLists(casadi_fns, cuda_name=casadi_fns_file_name)
    t_compile = time.time()
    compileCUDACode()
    t_compile = time.time() - t_compile
    print(f"Compilation time: {t_compile:.2f} seconds")


# Helper functions
def compileCUDACode():
    build_root = Path(CUSADI_ROOT_DIR)
    print("Compiling CUDA code..., build directory:", build_root)
    build_dir = build_root / "build"
    build_root.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["cmake", ".."], cwd=str(build_dir), check=True)
        subprocess.run(["make", "-j"], cwd=str(build_dir), check=True)
        print("Compilation complete.")
    except subprocess.CalledProcessError:
        print("Compilation failed.")
        exit(1)


def printParserArguments(parser, args):
    # Print out all arguments, descriptions, and default values in a formatted manner
    print(f"\n{'Argument':<20} {'Description':<80} {'Default':<10} {'Current Value':<10}")
    print("=" * 140)
    for action in parser._actions:
        if action.dest == "help":
            continue
        arg_strings = ", ".join(action.option_strings) if action.option_strings else action.dest
        description = action.help or "No description"
        default = action.default if action.default is not argparse.SUPPRESS else "No default"
        # Convert to string to avoid formatting None or other non-string types
        default_str = str(default)
        current_value = getattr(args, action.dest, default)
        current_value_str = str(current_value)
        print(f"{arg_strings:<20} {description:<80} {default_str:<10} {current_value_str:<10}")
    print()


def setupParser():
    parser = argparse.ArgumentParser(description="Script to generate parallelized code from CasADi functions")
    parser.add_argument("--fn", type=str, dest="fn_name", default="all", help='Function to parallelize in cusadi/casadi_functions, defaults to "all"')
    parser.add_argument(
        "--precision", type=bool, dest="precision", default=False, help="Precision of generated fn. True: double, False: float. Defaults to double"
    )
    parser.add_argument("--gen_CUDA", type=bool, dest="gen_CUDA", default=True, help="Generate CUDA codegen. Defaults to True")
    parser.add_argument(
        "--gen_pytorch", type=bool, dest="gen_pytorch", default=False, help="Generate Pytorch codegen in addition to CUDA. Defaults to False"
    )
    parser.add_argument(
        "--benchmark", type=bool, dest="codegen_benchmark_fns", default=False, help="Generate functions for benchmarking. Defaults to False"
    )
    parser.add_argument("--func_dir", type=str, dest="func_dir", default=None, help="Directory of CasADi functions to process")
    parser.add_argument("--clean_compile", type=bool, dest="clean_compile", default=False, help="Clean the build directory before compiling")
    return parser


if __name__ == "__main__":
    parser = setupParser()
    args = parser.parse_args()
    printParserArguments(parser, args)
    main(args)
