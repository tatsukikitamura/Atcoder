# How to use

## Basic (Quick start)

### Install packages

Check if you can use the following commands:
```sh
cmake --version
g++ --version
```

If some commands are not found, for Ubuntu users, you can use the following commands to install.
```sh
sudo apt install cmake g++
```

### Build and run

Run the following commands to build and run the solver, and confirm the results.
```sh
chmod +x build.sh # make sure build.sh is executable
./build.sh
./build/run-solver data/in-small-1.txt
cat data/out-small-1.txt
```

## Advanced (Target build)

### Install

Check if you can use the following commands:
```sh
x86_64-linux-gnu-g++ --version
```

If some commands are not found, for Ubuntu users, you can use the following commands to install.
```sh
sudo apt install g++-x86-64-linux-gnu
```

### Build

Run the following command to perform target build. We recommend using Intel SDE to emulate the program (Please check the contest guide).
```sh
./build.sh rocketlake
```

## Advanced (Dockerfile)

### Install

Dockerfile is also provided, for users who want to use the same compiler version as the actual environment.

- Run the following command, if you want to build an image with `g++-14.1.0` for your native architecture.
  ```sh
  docker build -t fixcon-cpu2024:latest ./compiler/pc-linux-gnu-gcc-14.1.0/
  ```

- Run the following command, if you want to build an image with `x86_64-linux-gnu-g++-14.1.0`.
  ```sh
  docker build -t fixcon-cpu2024-x86_64:latest ./compiler/x86_64-linux-gnu-gcc-14.1.0/
  ```

### Build and run

Run the following command to build and run the solver.
```sh
docker run --rm -v $(pwd):/work -u $(id -u):$(id -g) -w /work fixcon-cpu2024:latest \
bash -c "./build.sh && ./build/run-solver data/in-example.txt && cat data/out-example.txt"
```

Run the following command to perform target build. We recommend using Intel SDE to emulate the program (Please check the contest guide).
```sh
docker run --rm -v $(pwd):/work -u $(id -u):$(id -g) -w /work fixcon-cpu2024:latest \
bash -c "./build.sh rocketlake"
```
