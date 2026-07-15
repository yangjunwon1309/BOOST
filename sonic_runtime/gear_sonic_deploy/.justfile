# Like GNU `make`, but `just` rustier.
# https://just.systems/
# run `just` from this directory to see available commands

alias b := build
alias r := run
alias t := test
alias c := clean
alias ch := check

# Default command when 'just' is run without arguments
default:
  @just --list

# Get the number of cores
CORES := if os() == "macos" { `sysctl -n hw.ncpu` } else if os() == "linux" { `nproc` } else { "1" }

# Build the project
build *build_type='Release':
  @if [ -f build/CMakeCache.txt ] && ! grep -q "CMAKE_HOME_DIRECTORY:INTERNAL=$(pwd)" build/CMakeCache.txt; then echo "Existing CMake cache was created for a different source path; moving it aside."; mv build "build.stale.$(date +%s%N)"; fi
  @if [ -n "${TensorRT_ROOT:-}" ] && [ -f build/CMakeCache.txt ]; then expected="$(cd "$TensorRT_ROOT" && pwd)/include"; cached="$(sed -n 's/^TensorRT_INCLUDE_DIR:PATH=//p' build/CMakeCache.txt | head -n 1)"; if [ "$cached" != "$expected" ] || [ ! -f "$cached/NvInfer.h" ]; then echo "Existing CMake cache has stale TensorRT include: ${cached:-<unset>}; moving it aside."; mv build "build.stale.$(date +%s%N)"; fi; fi
  @if [ -f build/CMakeCache.txt ]; then tool="$(sed -n 's/^CMAKE_MAKE_PROGRAM:FILEPATH=//p' build/CMakeCache.txt | head -n 1)"; if [ -n "$tool" ] && [ "$tool" != "CMAKE_MAKE_PROGRAM-NOTFOUND" ] && [ ! -x "$tool" ]; then echo "Existing CMake cache points to missing build tool: $tool; moving it aside."; mv build "build.stale.$(date +%s%N)"; fi; fi
  @if [ -d build ] && [ ! -w build ]; then echo "Existing build directory is not writable by $(whoami); moving it aside."; mv build "build.stale.$(date +%s%N)"; fi
  @mkdir -p build target/release
  @echo "Configuring the build system..."
  @cd build && cmake -S .. -B . -DCMAKE_BUILD_TYPE={{build_type}} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ${TensorRT_ROOT:+-DTensorRT_ROOT=$TensorRT_ROOT}
  @echo "Building the project..."
  @cd build && cmake --build . -j{{CORES}}

# Run a package
run *package='hello':
  @./target/release/{{package}}

# Run code quality tools
test:
  @echo "Running tests..."

# Remove build artifacts and non-essential files
clean:
  @echo "Cleaning..."
  @rm -rf build
  @rm -rf target

# Run code quality tools
check:
  @echo "Running code quality tools..."
  @cppcheck --error-exitcode=1 --project=build/compile_commands.json -i build/_deps/
