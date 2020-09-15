/**
 * @file solve.cpp
 * @brief Definition of member functions regarding solving in AmgXSolver.
 * @author Pi-Yueh Chuang (pychuang@gwu.edu)
 * @date 2016-01-08
 * \copyright Copyright (c) 2015-2019 Pi-Yueh Chuang, Lorena A. Barba.
 *            This project is released under MIT License.
 */


// AmgXWrapper
# include "AmgXSolver.hpp"


/* \implements AmgXSolver::solve_real */
PetscErrorCode AmgXSolver::solve(Vec &p, Vec &b)
{
    PetscFunctionBeginUser;

    PetscErrorCode      ierr;

    double              *unks,
                        *rhs;

    int                 size;

    AMGX_SOLVE_STATUS   status;

    // get size of local vector (p and b should have the same local size)
    ierr = VecGetLocalSize(p, &size); CHK;

    // get pointers to the raw data of local vectors
    ierr = VecGetArray(p, &unks); CHK;
    ierr = VecGetArray(b, &rhs); CHK;

    // upload vectors to AmgX
    AMGX_vector_upload(AmgXP, size, 1, unks);
    AMGX_vector_upload(AmgXRHS, size, 1, rhs);

    // solve
    ierr = MPI_Barrier(world); CHK;
    AMGX_solver_solve(solver, AmgXRHS, AmgXP);

    // get the status of the solver
    AMGX_solver_get_status(solver, &status);

    // check whether the solver successfully solve the problem
    if (status != AMGX_SOLVE_SUCCESS) SETERRQ1(world,
            PETSC_ERR_CONV_FAILED, "AmgX solver failed to solve the system! "
            "The error code is %d.\n", status);

    // download data from device
    AMGX_vector_download(AmgXP, unks);

    // restore PETSc vectors
    ierr = VecRestoreArray(p, &unks); CHK;
    ierr = VecRestoreArray(b, &rhs); CHK;

    PetscFunctionReturn(0);
}
