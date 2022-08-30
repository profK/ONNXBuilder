# ONNXBuilder
A library for building ONNX models in memory with C++.
Has only been tested do far on Win64

## Usage

First clone this repo

Next, you will need the goolge protobuf runtime libraries.
The easiest way I ghave found to get those on windows is to use vcpkg.
If you do not yet have vcpkg, you can install it following the directions
at https://vcpkg.io/en/getting-started.html
Once you have vcpkg installed you can get the protobuf libraries by using
the command:  vcpkg install protobuf


