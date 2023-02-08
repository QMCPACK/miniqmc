# Optimisation
## ajout de flags de compilation
```bash
	cmake  -DCMAKE_BUILD_TYPE=None -DCMAKE_CXX_COMPILER=mpicxx CMAKE_CXX_FLAGS="-Ofast -march=native -finline-functions -funroll-loops -ftree-loop-vectorize -ftree-vectorize" ..  

```
Expliquer les flags