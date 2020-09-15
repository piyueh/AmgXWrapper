/**
 * \file init.cpp
 * \brief Definition of some member functions of the class AmgXSolver.
 * \author Pi-Yueh Chuang (pychuang@gwu.edu)
 * \date 2016-01-08
 * \copyright Copyright (c) 2015-2019 Pi-Yueh Chuang, Lorena A. Barba.
 *            This project is released under MIT License.
 */


// CUDA
# include <cuda_runtime.h>

// AmgXSolver
# include "AmgXSolver.hpp"


/* \implements AmgXSolver::AmgXSolver */
AmgXSolver::AmgXSolver(const MPI_Comm &comm,
        const std::string &modeStr, const std::string &cfgFile)
{
    initialize(comm, modeStr, cfgFile);
}


/* \implements AmgXSolver::~AmgXSolver */
AmgXSolver::~AmgXSolver()
{
    if (isInitialized) finalize();
}


/* \implements AmgXSolver::initialize */
PetscErrorCode AmgXSolver::initialize(const MPI_Comm &comm,
        const std::string &modeStr, const std::string &cfgFile)
{
    PetscErrorCode      ierr;

    PetscFunctionBeginUser;

    // if this instance has already been initialized, skip
    if (isInitialized) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
            "This AmgXSolver instance has been initialized on this process.");

    // increase the number of AmgXSolver instances
    count += 1;

    // get the name of this node
    int     len;
    char    name[MPI_MAX_PROCESSOR_NAME];
    ierr = MPI_Get_processor_name(name, &len); CHK;
    nodeName = name;

    // get the mode of AmgX solver
    ierr = setMode(modeStr); CHK;

    // initialize communicators and corresponding information
    ierr = initMPIcomms(comm); CHK;

    // only processes in gpuWorld are required to initialize AmgX
    ierr = initAmgX(cfgFile); CHK;

    // a bool indicating if this instance is initialized
    isInitialized = true;

    PetscFunctionReturn(0);
}


/* \implements AmgXSolver::initMPIcomms */
PetscErrorCode AmgXSolver::initMPIcomms(const MPI_Comm &comm)
{
    PetscErrorCode      ierr;
    PetscMPIInt         nDevs;
    PetscMPIInt         nBasic, nRemain;

    PetscFunctionBeginUser;

    // get the number of total cuda devices
    CHECK(cudaGetDeviceCount(&nDevs));

    // Check whether there is at least one CUDA device on this node
    if (nDevs == 0) SETERRQ1(MPI_COMM_WORLD, PETSC_ERR_SUP_SYS,
            "There is no CUDA device on the node %s !\n", nodeName.c_str());

    // duplicate the global communicator
    ierr = MPI_Comm_dup(comm, &world); CHK;
    ierr = MPI_Comm_set_name(world, "AmgXWrold"); CHK;

    // get size and rank for global communicator
    ierr = MPI_Comm_size(world, &world_size); CHK;
    ierr = MPI_Comm_rank(world, &world_rank); CHK;

    // set up corresponding ID of the device used by each local process
    nBasic = world_size / nDevs;
    nRemain = world_size % nDevs;
    if (world_rank < (nBasic+1)*nRemain) {
        devID = world_rank / (nBasic + 1);
    }
    else {
        devID = (world_rank - (nBasic+1)*nRemain) / nBasic + nRemain;
    }
    ierr = MPI_Barrier(world); CHK;

    return 0;
}


/* \implements AmgXSolver::initAmgX */
PetscErrorCode AmgXSolver::initAmgX(const std::string &cfgFile)
{
    PetscFunctionBeginUser;

    // only the first instance (AmgX solver) is in charge of initializing AmgX
    if (count == 1)
    {
        // initialize AmgX
        AMGX_SAFE_CALL(AMGX_initialize());

        // intialize AmgX plugings
        AMGX_SAFE_CALL(AMGX_initialize_plugins());

        // only the master process can output something on the screen
        AMGX_SAFE_CALL(AMGX_register_print_callback(
                    [](const char *msg, int length)->void
                    {PetscPrintf(PETSC_COMM_WORLD, "%s", msg);}));

        // let AmgX to handle errors returned
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
    }

    // create an AmgX configure object
    AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, cfgFile.c_str()));

    // let AmgX handle returned error codes internally
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));

    // create an AmgX resource object, only the first instance is in charge
    if (count == 1) AMGX_resources_create(&rsrc, cfg, &world, 1, &devID);

    // create AmgX vector object for unknowns and RHS
    AMGX_vector_create(&AmgXP, rsrc, mode);
    AMGX_vector_create(&AmgXRHS, rsrc, mode);

    // create AmgX matrix object for unknowns and RHS
    AMGX_matrix_create(&AmgXA, rsrc, mode);

    // create an AmgX solver object
    AMGX_solver_create(&solver, rsrc, mode, cfg);

    // obtain the default number of rings based on current configuration
    AMGX_config_get_default_number_of_rings(cfg, &ring);

    PetscFunctionReturn(0);
}


/* \implements AmgXSolver::finalize */
PetscErrorCode AmgXSolver::finalize()
{
    PetscErrorCode      ierr;

    PetscFunctionBeginUser;

    // skip if this instance has not been initialized
    if (! isInitialized)
    {
        ierr = PetscPrintf(PETSC_COMM_WORLD,
                "This AmgXWrapper has not been initialized. "
                "Please initialize it before finalization.\n"); CHK;

        PetscFunctionReturn(0);
    }

    // destroy solver instance
    AMGX_solver_destroy(solver);

    // destroy matrix instance
    AMGX_matrix_destroy(AmgXA);

    // destroy RHS and unknown vectors
    AMGX_vector_destroy(AmgXP);
    AMGX_vector_destroy(AmgXRHS);

    // only the last instance need to destroy resource and finalizing AmgX
    if (count == 1)
    {
        AMGX_resources_destroy(rsrc);
        AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

        AMGX_SAFE_CALL(AMGX_finalize_plugins());
        AMGX_SAFE_CALL(AMGX_finalize());
    }
    else
    {
        AMGX_config_destroy(cfg);
    }

    // re-set necessary variables in case users want to reuse
    // the variable of this instance for a new instance
    ierr = MPI_Comm_free(&world); CHK;

    // decrease the number of instances
    count -= 1;

    // change status
    isInitialized = false;

    PetscFunctionReturn(0);
}
