#!/bin/bash
DIM_LIST=(32 64 128 256 512 1024 2048 4096 8192)
NP_LIST=(1 2 4 8)
DATA_TYPE=("uint8_t" "uint16_t")

# Configuration

BUILD_DIR="build"
EXEC_NAME="conways-game-2D"
OUTPUT_CSV="benchmark_results_conways2D.csv"

# Initialize CSV with headers
echo "Grid_Dim,Data_Type,MPI_Ranks,Time_Seconds" > "${OUTPUT_CSV}"

for dtype in "${DATA_TYPE[@]}"; do
  for np in "${NP_LIST[@]}"; do
    for dim in "${DIM_LIST[@]}"; do
      echo " [↖(^ω^)↗] Configuring: DIM=${dim}*${dim}, NP=${np}"

      cmake -B "${BUILD_DIR}" \
            -DCMAKE_BUILD_TYPE=Release \
            -DDIM_X="${dim}" \
            -DDIM_Y="${dim}" \
            -DDATA_TYPE="${dtype}" \
            > /dev/null 2>&1

      if [ $? -ne 0 ]; then
        echo " [!(~_~;?] CMake configuration failed. Continuing."
        continue
      fi

      cmake --build "${BUILD_DIR}" \
            --target "${EXEC_NAME}" \
            -j 4 \
            > /dev/null 2>&1

      if [ $? -ne 0 ]; then
        echo " [!(~_~;?] CMake build failed. Continuing."
        continue
      fi

      echo " [V(^_^)V] Running mpiexec -np ${np} ..."

      RAW_OUTPUT=$(mpiexec -np "${np}" ./"${BUILD_DIR}"/"${EXEC_NAME}" 2>&1)
      EVOLVE_TIME=$(echo "$RAW_OUTPUT" | grep -i "Total evolving time:" | awk '{print $4}')

      if [ -z "$EVOLVE_TIME" ]; then
        echo " [!(~_~;?] Failed to extract time. Raw output:"
        echo "$RAW_OUTPUT"
        EVOLVE_TIME="N/A"
      else
        echo " [+(^ω^)+] Evolve Time: ${EVOLVE_TIME} s"
      fi    

      echo "${dim},${dtype},${np},${EVOLVE_TIME}" >> "${OUTPUT_CSV}"

    done
  done
done

echo "+--------------------------------------------------------+"
echo " [♪───(≧∇≦)────♪] Benchmark Complete."
echo " Results saved to ${OUTPUT_CSV}"
echo "+--------------------------------------------------------+"