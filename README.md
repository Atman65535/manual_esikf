# Manual ESIKF

## About ROS workspace
- I want to build ikd tree, fastlio by myself.

# First Attempt: V1
This repository contains two parts:
1. `slam_hardware`, a stm32 cubeMX project for lower machine.
2. `slam_software`, pure cpp project for upper level algorithm.


## `slam_hardware`

1. make sure your computer contains:
  - `arm-gcc-none-eabi` for build
  - `openocd` for load program on your chip
  - `gdb-multiarch` for dbg
  - `tio` for serial monitoring.

2. `cd slam_hardware && mkdir build`
3. `cd build && cmake ..`
4. `make slam_hardware -j`
5. `openocd -f interface/stlink.cfg -f target/stm32f4x.cfg -c "program slam_hardware.elf reset"`

That's all

