# miniQMC - QMCPACK Miniapp

## goals for v0.1
1. Jastrow 2-body kernel, distance tables kernel(s), 1D bspline.
2. 3D bspline kernel.
3. easy to use self-checking capability (compile-time switch okay, runtime is nicer)
4. timings for the individual kernels (not necessarily nested, not switchable - always on)
5. shrink the codebase as much as possible - ONLY code necessary to run the kernels of interest
6. inverse update kernel
7. Jastrow 1-body, 3-body(e-e-I) kernels
8. high-level (physics) description for each kernel (we can probably recruit help for this)
9. doxygen comments in the code

# General Principles

## Short term goals:
1. Evaluate programming models for performance portability
2. Explore alternative algorithms and data structures for critical kernels
3. Collaborate with non-QMCPACK developers, e.g. vendors and ECP-ST projects
4. Solve reintegration issues within miniapp before bringing new developments
   back to QMCPACK
5. Make an initial release/handoff to OpenMP and Kokkos assessors quickly
   enough such that the ECP milestones will be delivered by end September 2017.

## Long term goals
1. Evolve the capabilities iteratively over a series of releases
2. Continue to explore new language features, libraries, and high-risk ideas
   eficiently
3. Benchmark new hardware/software platforms.
4. Serve as a new level of tests above the unit tests.
5. Collaborate with non-QMCPACK developers
6. Easier training for new project members and students

## Requirements
1. As simple to use as possible to ensure maximum productivity
    1. Simple build system based on flat Makefiles
    2. Reasonable default options and easy to use command-line interface - no
       input files for easier scriptability
    3. Self timing with breakdown by kernel
        1. Output in easily parseable text format, e.g. gnuplot columns or csv
    4. No large data or auxiliary files

2. As simple to develop as possible to ensure maximum productivity
    1. Commented source code, doxygen headers on functions.
    2. Completely internal self-checking for most efficient development workflow
        1. Can be switched on/off at runtime
        2. When switched off, no residual effects on timings or memory usage
    3. Minimal/No library dependency
        1. To keep building/maintenance as simple as possible
        2. BLAS is okay
        3. Avoid HDF5, FFTW, BOOST, libXML etc.
        4. No MPI parallelization
    4. As small as possible - only has the code necessary to run the kernels of
       interest
    5. Minimal functionality, e.g. only value and combined val-grad-lap for
       wavefunction components.
    6. Kernel orthogonality
        1. Each is hackable/tunable without getting into details of the others
        2. This is required for novel hardware and software technology
          assessments
        3. Keep significant implementation out of the coupling/shared code
        4. Driver and shared data structures rarely/never need changes
        5. Easy for newcomers to work on isolated parts of the code
    7.  C++ productivity without over-engineering
        1. Abstractions hide data and implementation for safety and convenience
        2. Loosely coupled instead of deep hierarchies for hackability
2. Communicates our intentions and needs from C++, libraries, etc.
    1. Uses C++ features - write the code how we would like to see it
    2. Best possible algorithm for computational and memory footprint scaling
       but easy to understand implementation without optimization
        1.  Main entry point for new collaborators
        2.  Starting point for exploring optimizations
    3.  No pre-optimization or lowered abstractions
        1.  Make the empirical data tell us where to compromise
3. Generally represents QMCPACK OOP design and QMC algorithms
    1. We have an accessible enough app that *little/no supervision* is required
       to use it
    2. Must ensure maximum flexibility for new algorithms and data structures
    3. Have necessary physics abstractions for functional flexibility
        1. Wavefunction components (det, J1/2/3). 
        2. Boundary conditions (Open, PPP, PPN, PNN). Initially fully periodic
          only (PPP).
        3. Numerical functor (Bspline, Polynomial). Only spline J1J2 and
          polynomial J3 in first version, no functor.
    4. Start from QMCPACK API, but simplify as much as possible
        1. Can break consistency if necessary (minimize if possible)
    5. During miniapp development, no extra effort necessary due to
       reintegration concerns
    6. Use the same call sequence as real simulations
        1. Enables interprocedural algorithmic exploration 
1. Flexible to change benchmark system and system size
    1. Only ECP-NiO, CORAL-graphite
    2. From ~100 to ~10k electrons sizes
    3. Via command line
2. Minimal MPI - no MPI-based parallelization
    1. Only minimum MPI present for tools/job wrappers to interface (e.g.
       MPI_Init()/Finalize())
    2. No introduction of additional build dependency on MPI
3. Documentation
    1. Need a high-level, non-code-specific explanation of kernels/algorithms
        1. Governing equations, operations, etc.
    2. README included with source code
        1. How to run, check, time, etc.
        2. How to scale inputs
        3. Small: single process
        4. Medium: single gpu, node, etc.
        5. Big: rack
        6. Huge: challenge problem
    3. List of compilers and platforms we’ve checked
    4. What is in/out of scope for optimization
        1. No optimization into corners, magic numbers, etc.
        2. Specifics about physical realities, restrictions, etc.
    5. Contact info: support, patches, changes, etc.
