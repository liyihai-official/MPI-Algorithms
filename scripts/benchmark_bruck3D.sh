#!/bin/bash
DIM_LIST=(16 32 64 128 256 512 1024 2048 4096)
NP_LIST=(2 4 8)
DATA_TYPE=("uint8_t" "uint16_t")

# Configuration

BUILD_DIR="build"
EXEC_NAME="bruck_Alltoall"
OUTPUT_CSV="benchmark_results_bruck3D.csv"

# Initialize CSV with headers
echo "Grid_Dim,Data_Type,MPI_Ranks,Bruck_Time_Seconds, MPI_Time_Seconds" > "${OUTPUT_CSV}"

for dtype in "${DATA_TYPE[@]}"; do
  for np in "${NP_LIST[@]}"; do
    for dim in "${DIM_LIST[@]}"; do
      echo " [↖(^ω^)↗] Configuring: DIM=${dim}*${dim}, NP=${np}"

      cmake -B "${BUILD_DIR}" \
            -DCMAKE_BUILD_TYPE=Release \
            -DDIM_X="${dim}" \
            -DDIM_Y="${dim}" \
            -DDIM_Z="${dim}" \
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
      BRUCK_EVOLVE_TIME=$(echo "$RAW_OUTPUT" | grep -i "Bruck avg. time consumption:" | awk '{print $5}')
      MPI_EVOLVE_TIME=$(echo "$RAW_OUTPUT" | grep -i "MPI_Alltoall avg. time consumption:" | awk '{print $5}')

      if [ -z "$BRUCK_EVOLVE_TIME" ]; then
        echo " [!(~_~;?] Failed to extract time. Raw output:"
        echo "$RAW_OUTPUT"
        BRUCK_EVOLVE_TIME="N/A"
      else
        echo " [+(^ω^)+] Bruck Time: ${BRUCK_EVOLVE_TIME} s, MPI Time: ${MPI_EVOLVE_TIME}"
      fi    

      echo "${dim},${dtype},${np},${BRUCK_EVOLVE_TIME}, ${MPI_EVOLVE_TIME}" >> "${OUTPUT_CSV}"

    done
  done
done

echo "+--------------------------------------------------------+"
echo " [♪───(≧∇≦)────♪] Benchmark Complete."
echo " Results saved to ${OUTPUT_CSV}"
echo "+--------------------------------------------------------+"