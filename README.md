# ONNXBuilder
A library for building ONNX models in memory with C++.
Has only been tested do far on Win64

## Usage

First clone this repo

Next, you will need the google protobuf iinclude files and runtime libraries.
The easiest way I have found to get those on windows is to use vcpkg.
If you do not yet have vcpkg, you can install it following the directions
at https://vcpkg.io/en/getting-started.html

Remember to run vcpkg integrate install to enable vcpkg in VS2022 and Rider2022

Once you have vcpkg installed you can get the protobuf libraries by using
the command:  vcpkg install protobuf

## What can go wrong
Remember to set the library and test builds both to release.  Failure to do so can
cause  link errors/



