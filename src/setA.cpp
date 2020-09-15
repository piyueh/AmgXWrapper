/**
 * \file setA.cpp
 * \brief Definition of member functions regarding setting A in AmgXSolver.
 * \author Pi-Yueh Chuang (pychuang@gwu.edu)
 * \date 2016-01-08
 * \copyright Copyright (c) 2015-2019 Pi-Yueh Chuang, Lorena A. Barba.
 * \copyright Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *            This project is released under MIT License.
 */


// STD
# include <cstring>
# include <algorithm>

// AmgXSolver
# include "AmgXSolver.hpp"


/* \implements AmgXSolver::setA */
PetscErrorCode AmgXSolver::setA(const Mat &A)
{
    PetscFunctionBeginUser;

    PetscErrorCode      ierr;

    Mat                 localA;

    IS                  devIS;

    PetscInt            nGlobalRows,
                        nLocalRows;
    PetscBool           usesOffsets;

    std::vector<PetscInt>       row;
    std::vector<PetscInt64>     col;
    std::vector<PetscScalar>    data;
    std::vector<PetscInt>       partData;


    // get number of rows in global matrix
    ierr = MatGetSize(A, &nGlobalRows, nullptr); CHK;

    // get the row indices of redistributed matrix owned by processes in gpuWorld
    ierr = getDevIS(A, devIS); CHK;

    // get sequential local portion of redistributed matrix
    ierr = getLocalA(A, devIS, localA); CHK;

    // get compressed row layout of the local Mat
    ierr = getLocalMatRawData(localA, nLocalRows, row, col, data); CHK;

    // destroy local matrix
    ierr = destroyLocalA(A, localA); CHK;

    // get a partition vector required by AmgX
    ierr = getPartData(devIS, nGlobalRows, partData, usesOffsets); CHK;


    // upload matrix A to AmgX
    if (world != MPI_COMM_NULL)
    {
        ierr = MPI_Barrier(world); CHK;
        // offsets need to be 64 bit, since we use 64 bit column indices
        std::vector<PetscInt64> offsets;

        AMGX_distribution_handle dist;
        AMGX_distribution_create(&dist, cfg);
        if (usesOffsets) {
            offsets.assign(partData.begin(), partData.end());
            AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS, offsets.data());
        } else {
            AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_VECTOR, partData.data());
        }
        AMGX_matrix_upload_distributed(
                AmgXA, nGlobalRows, nLocalRows, row[nLocalRows],
                1, 1, row.data(), col.data(), data.data(),
                nullptr, dist);
        AMGX_distribution_destroy(dist);

        // bind the matrix A to the solver
        ierr = MPI_Barrier(world); CHK;
        AMGX_solver_setup(solver, AmgXA);

        // connect (bind) vectors to the matrix
        AMGX_vector_bind(AmgXP, AmgXA);
        AMGX_vector_bind(AmgXRHS, AmgXA);
    }
    ierr = MPI_Barrier(world); CHK;

    // destroy temporary PETSc objects
    ierr = ISDestroy(&devIS); CHK;

    PetscFunctionReturn(0);
}


/* \implements AmgXSolver::getDevIS */
PetscErrorCode AmgXSolver::getDevIS(const Mat &A, IS &devIS)
{
    PetscFunctionBeginUser;

    PetscErrorCode      ierr;
    IS                  tempIS;

    // get index sets of A locally owned by each process
    // note that devIS is now a serial IS on each process
    ierr = MatGetOwnershipIS(A, &devIS, nullptr); CHK;

    // devIS is not guaranteed to be sorted. We sort it here.
    ierr = ISSort(devIS); CHK;

    PetscFunctionReturn(0);
}


/* \implements AmgXSolver::getLocalA */
PetscErrorCode AmgXSolver::getLocalA(const Mat &A, const IS &devIS, Mat &localA)
{
    PetscFunctionBeginUser;

    PetscErrorCode      ierr;
    MatType             type;

    // get the Mat type
    ierr = MatGetType(A, &type); CHK;

    // check whether the Mat type is supported
    if (std::strcmp(type, MATSEQAIJ) == 0) // sequential AIJ
    {
        // make localA point to the same memory space as A does
        localA = A;
    }
    else if (std::strcmp(type, MATMPIAIJ) == 0)
    {
        Mat                 tempA;

        // redistribute matrix and also get corresponding scatters.
        ierr = redistMat(A, devIS, tempA); CHK;

        // get local matrix from redistributed matrix
        ierr = MatMPIAIJGetLocalMat(tempA, MAT_INITIAL_MATRIX, &localA); CHK;

        // destroy redistributed matrix
        if (tempA == A)
        {
            tempA = nullptr;
        }
        else
        {
            ierr = MatDestroy(&tempA); CHK;
        }
    }
    else
    {
        SETERRQ1(world, PETSC_ERR_ARG_WRONG,
                "Mat type %s is not supported!\n", type);
    }

    PetscFunctionReturn(0);
}


/* \implements AmgXSolver::redistMat */
PetscErrorCode AmgXSolver::redistMat(const Mat &A, const IS &devIS, Mat &newA)
{
    PetscFunctionBeginUser;

    PetscErrorCode      ierr;

    newA = A;

    PetscFunctionReturn(0);
}


/* \implements AmgXSolver::getLocalMatRawData */
PetscErrorCode AmgXSolver::getLocalMatRawData(const Mat &localA,
        PetscInt &localN, std::vector<PetscInt> &row,
        std::vector<PetscInt64> &col, std::vector<PetscScalar> &data)
{
    PetscFunctionBeginUser;

    PetscErrorCode      ierr;

    const PetscInt      *rawCol,
                        *rawRow;

    PetscScalar         *rawData;

    PetscInt            rawN;

    PetscBool           done;

    // get row and column indices in compressed row format
    ierr = MatGetRowIJ(localA, 0, PETSC_FALSE, PETSC_FALSE,
            &rawN, &rawRow, &rawCol, &done); CHK;

    // rawN will be returned by MatRestoreRowIJ, so we have to copy it
    localN = rawN;

    // check if the function worked
    if (! done)
        SETERRQ(world, PETSC_ERR_SIG, "MatGetRowIJ did not work!");

    // get data
    ierr = MatSeqAIJGetArray(localA, &rawData); CHK;

    // copy values to STL vector. Note: there is an implicit conversion from
    // PetscInt to PetscInt64 for the column vector
    col.assign(rawCol, rawCol+rawRow[localN]);
    row.assign(rawRow, rawRow+localN+1);
    data.assign(rawData, rawData+rawRow[localN]);


    // return ownership of memory space to PETSc
    ierr = MatRestoreRowIJ(localA, 0, PETSC_FALSE, PETSC_FALSE,
            &rawN, &rawRow, &rawCol, &done); CHK;

    // check if the function worked
    if (! done)
        SETERRQ(world, PETSC_ERR_SIG, "MatRestoreRowIJ did not work!");

    // return ownership of memory space to PETSc
    ierr = MatSeqAIJRestoreArray(localA, &rawData); CHK;

    PetscFunctionReturn(0);
}


/* \implements AmgXSolver::destroyLocalA */
PetscErrorCode AmgXSolver::destroyLocalA(const Mat &A, Mat &localA)
{
    PetscFunctionBeginUser;

    PetscErrorCode      ierr;

    MatType             type;

    // Get the Mat type
    ierr = MatGetType(A, &type); CHK;

    // when A is sequential, we can not destroy the memory space
    if (std::strcmp(type, MATSEQAIJ) == 0)
    {
        localA = nullptr;
    }
    // for parallel case, localA can be safely destroyed
    else if (std::strcmp(type, MATMPIAIJ) == 0)
    {
        ierr = MatDestroy(&localA); CHK;
    }

    PetscFunctionReturn(0);
}


/* \implements AmgXSolver::getPartData */
PetscErrorCode AmgXSolver::getPartData(
        const IS &devIS, const PetscInt &N, std::vector<PetscInt> &partData, PetscBool &usesOffsets)
{
    PetscFunctionBeginUser;

    PetscErrorCode      ierr;

    VecScatter          scatter;
    Vec                 tempMPI,
                        tempSEQ;

    PetscInt            n;
    PetscScalar         *tempPartVec;

    ierr = ISGetLocalSize(devIS, &n); CHK;

    if (world != MPI_COMM_NULL)
    {
        // check if sorted/contiguous, then we can skip expensive scatters
        checkForContiguousPartitioning(devIS, usesOffsets, partData);
        if (!usesOffsets)
        {
            ierr = VecCreateMPI(world, n, N, &tempMPI); CHK;

            IS      is;
            ierr = ISOnComm(devIS, world, PETSC_USE_POINTER, &is); CHK;
            ierr = VecISSet(tempMPI, is, (PetscScalar) world_rank); CHK;
            ierr = ISDestroy(&is); CHK;

            ierr = VecScatterCreateToAll(tempMPI, &scatter, &tempSEQ); CHK;
            ierr = VecScatterBegin(scatter,
                    tempMPI, tempSEQ, INSERT_VALUES, SCATTER_FORWARD); CHK;
            ierr = VecScatterEnd(scatter,
                    tempMPI, tempSEQ, INSERT_VALUES, SCATTER_FORWARD); CHK;
            ierr = VecScatterDestroy(&scatter); CHK;
            ierr = VecDestroy(&tempMPI); CHK;

            ierr = VecGetArray(tempSEQ, &tempPartVec); CHK;

            partData.assign(tempPartVec, tempPartVec+N);

            ierr = VecRestoreArray(tempSEQ, &tempPartVec); CHK;

            ierr = VecDestroy(&tempSEQ); CHK;
        }
    }
    ierr = MPI_Barrier(world); CHK;

    PetscFunctionReturn(0);
}


/* \implements AmgXSolver::checkForContiguousPartitioning */
PetscErrorCode AmgXSolver::checkForContiguousPartitioning(
    const IS &devIS, PetscBool &isContiguous, std::vector<PetscInt> &partOffsets)
{
    PetscFunctionBeginUser;
    PetscErrorCode      ierr;
    PetscBool sorted;
    PetscInt ismax= -2; // marker for "unsorted", allows to check after global sort

    ierr = ISSorted(devIS, &sorted); CHK;
    if (sorted)
    {
        ierr = ISGetMinMax(devIS, NULL, &ismax); CHK;
    }
    partOffsets.resize(world_size);
    ++ismax; // add 1 to allow reusing gathered ismax values as partition offsets for AMGX
    MPI_Allgather(&ismax, 1, MPIU_INT, &partOffsets[0], 1, MPIU_INT, world);
    bool all_sorted = std::is_sorted(partOffsets.begin(), partOffsets.end())
                        && partOffsets[0] != -1;
    if (all_sorted)
    {
        partOffsets.insert(partOffsets.begin(), 0); // partition 0 always starts at 0
        isContiguous = PETSC_TRUE;
    }
    else
    {
        isContiguous = PETSC_FALSE;
    }
    PetscFunctionReturn(0);
}
