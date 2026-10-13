#!/usr/bin/env python
import subprocess, sys, re
import os, glob
import time, datetime
import csv, json
import asyncio, shlex
import argparse, textwrap

# Function to run shell commands and capture output
def run_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return result.stdout.strip()

def run_cmd(cmd):
    async def inner():
        proc = await asyncio.create_subprocess_shell(
            shlex.join(cmd), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        return stdout.decode('utf-8', 'backslashreplace')

    return asyncio.run(inner())

GPU_MODES = ("a100", "mig", "l40")
MIG_TYPES = ("10g", "20g", "40g")

def require_gpu_mode(test, sub_option, mig_type):
    """Validate the mode / mig_type positionals shared by the GPU tests."""
    if sub_option not in GPU_MODES:
        sys.exit(f"{test} requires a mode: a100, mig, or l40\n"
                 f"  Usage: python maint_main.py {test} a100|mig|l40")

    if sub_option == "mig" and mig_type not in MIG_TYPES:
        sys.exit(f"{test} mig requires a mig_type: 10g, 20g, or 40g\n"
                 f"  Usage: python maint_main.py {test} mig 10g|20g|40g")

def enter_dir(*parts, clean=False):
    """Create the test's working directory (one level per part) and change into it.

    clean removes a directory left by a previous run first, so results from the
    two runs do not end up mixed together.
    """
    for part in parts:
        if not part:
            continue

        # These names are built from command line arguments, so make sure they
        # stay a plain directory name before anything gets removed
        if os.sep in part or part in ('.', '..'):
            sys.exit(f"Refusing to use '{part}' as a directory name")

        if clean and os.path.isdir(part):
            print(f"Removing previous {os.path.join(os.getcwd(), part)}")
            run_command(f"rm -rf {shlex.quote(part)}")

        os.makedirs(part, exist_ok=True)
        os.chdir(part)

    print(f"Created and changed directory to: {os.getcwd()}\n")
    return os.getcwd()

def submit_job(script="maint_job.slurm"):
    """Submit a Slurm script, exiting with sbatch's own error message if submission fails."""
    result = subprocess.run(f"sbatch {script}", shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0 or not stdout:
        sys.exit(f"sbatch failed to submit {script} (exit code {result.returncode}):\n"
                 f"{stderr or 'sbatch produced no output'}")

    if stderr:
        print(stderr)

    return stdout

def get_nodes(partition):
    print(f"Running tests in '{partition}' partition\n")
    # Extract the nodelist where STATE is "maint"
    nodelist_output = run_command(f"sinfo -p {partition} | awk '$5 == \"maint\" {{print $NF}}'")

    # Use scontrol to expand the nodelist and store each node in a list
    expanded_nodelist_output = run_command(f"scontrol show hostname {nodelist_output}")
    expanded_nodelist = expanded_nodelist_output.splitlines()

    i = 0

    for node in expanded_nodelist:
        i += 1

    # Store the final value of i
    nodes = i

    if nodes == 0:
        sys.exit(f'No available nodes in the "{partition}" partition for testing at the moment. Please check back later. Exiting.')
    else:
        print(f'Total number of Nodes in "{partition}" partition: ',nodes)

    return nodes

def get_gpu_nodes(partition):
    print(f"Running tests in '{partition}' partition\n")
    # Get node name, state, and GRES for each node in the partition
    node_info = run_command(f"sinfo -p {partition} -N -o \"%N %T %G\" | awk '$2 == \"maint\"'")

    a100_nodes = []
    mig_nodes = []
    l40_nodes = []

    for line in node_info.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        node = parts[0]
        gres = parts[2]

        # MIG GRES contain profile patterns like "a100_10g", "a100_20g", "a100_40g", etc.
        if re.search(r'a100_\d+', gres):
            mig_nodes.append(node)
        elif re.search(r'l40', gres, re.IGNORECASE):
            l40_nodes.append(node)
        else:
            a100_nodes.append(node)

    a100_count = len(a100_nodes)
    mig_count = len(mig_nodes)
    l40_count = len(l40_nodes)

    if a100_count == 0 and mig_count == 0 and l40_count == 0:
        sys.exit(f'No available nodes in the "{partition}" partition for testing at the moment. Please check back later. Exiting.')

    print(f'A100 Nodes in "{partition}" partition: {a100_count}')
    print(f'MIG Nodes in "{partition}" partition: {mig_count}')
    print(f'L40 Nodes in "{partition}" partition: {l40_count}')

    return a100_count, mig_count, l40_count

def get_earliest_reservation(res):
    lines = res.splitlines()  # Split the result into lines
    min_reservation = None
    min_start_time = None

    for line in lines:
        # Extract ReservationName and StartTime from each line
        if line.startswith("ReservationName"):
            reservation_name = line.split('=')[1].strip()
            reservation_name = reservation_name.replace("StartTime", "").strip()
            start_time = line.split('=')[2].strip()
            start_time = start_time.replace("EndTime", "").strip()

            # Compare the StartTime to find the earliest
            if min_start_time is None or start_time < min_start_time:
                min_start_time = start_time
                min_reservation = reservation_name

    if min_reservation:
        print(f"Earliest reservation: {min_reservation} with start time: {min_start_time}")
    else:
        print("No reservations found")

    return min_reservation

def resolve_reservation(reservation=None):
    """Return the reservation to use: the user supplied one (-r) if given, else the earliest."""
    res_output = run_command("scontrol show res")

    if reservation:
        available = [line.split('=')[1].replace("StartTime", "").strip()
                     for line in res_output.splitlines() if line.startswith("ReservationName")]
        if reservation not in available:
            sys.exit(f'Reservation "{reservation}" not found. Available reservations: '
                     + (", ".join(available) if available else "none"))
        print(f"Using user supplied reservation: {reservation}")
        return reservation

    return get_earliest_reservation(res_output)

def wait_for_file(pattern, check_interval=300, timeout=1800):
    """Waits for a file matching the pattern to appear, giving up after timeout seconds."""
    pattern = os.path.join(os.getcwd(), pattern)
    waited = 0

    while True:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            print(f"\nFile found: {matches[-1]}")
            return matches[-1]

        if timeout is not None and waited >= timeout:
            print(f"\nGave up waiting for {pattern} after {waited // 60} minutes")
            return None

        print(f"\nNo file matching {pattern} yet. Retrying in {check_interval // 60} minutes...")
        time.sleep(check_interval)
        waited += check_interval
            
class openfoam:
    def run(self, reservation=None):
        nodes = get_nodes("general")
        
        res = resolve_reservation(reservation)
        print("Using Reservation: ", res)
        
        # Create the openfoam directory and change into it
        enter_dir('openfoam')

        dir = 'motorBike'
        
        print("Cleaning up existing files and direcotries\n")
        
        del_dir = run_command("rm -rf *.err *.out motorBike")
        
        # Create the OFTOOL script
        oftools = textwrap.dedent("""\
        #!/bin/bash
        #------------------------------------------------------------------------
        RUNA()
        {{
        echo "## $1 :"`date`; $* >log.$1
        }}
        RUNP()
        {{
        echo "## $1 :"`date`; $MPIRUN $* -parallel >log.$1
        }}
        DONE()
        {{
            echo "## Done  :"`date`
            solv_time=$(grep ExecutionTime log.$1 | tail -n 1)
            TS=$(grep "Time = " log.$1 | grep -v Clock| tail -1| awk '{{print $3 }}')
            echo "OpenFOAM: solve_time=$solv_time   sim_time=$TS"
            rm -fr p*
        }}
        DCOMP()
        {{
            sdp=system/decomposeParDict-random
            if  [ $# -gt 0 ] ; then
              sdp=$1
            fi
            if [ ! -f "$sdp" ]; then
              cp system/decomposeParDict "$sdp"
            fi
            sed "/^numberOfSubdomains/ c \numberOfSubdomains $NP;" -i $sdp
            sed "/^method/c\method          scotch;"               -i $sdp
            echo "changingdecomp " $NP
            mv $sdp system/decomposeParDict
        }}
        ENDTIME()
        {{
        sed "/^endTime/c\\endTime  $1;"                     -i system/controlDict
        sed "/^writeInterval/c\\writeInterval  $2;"         -i system/controlDict
        }}
        
        """)
        #oftools = textwrap.dedent(oftools)
        # Write the SLURM script to a file
        with open("OFTOOLS", "w") as file:
            file.write(oftools)
            
        print("Writing the OFTOOLS script")
        
        # Define the SLURM script with placeholders for variables
        slurm_script = f"""\
        #!/bin/bash
        #SBATCH --job-name=maint_job       # Job name
        #SBATCH --output=%x.%j.out         # Output file
        #SBATCH --error=%x.%j.err          # Error log file
        #SBATCH --time=05:00:00            # Wall clock limit (1 hour)
        #SBATCH --nodes={nodes}            # Number of nodes (dynamic)
        #SBATCH --ntasks=1024              # Number of tasks (processes)
        #SBATCH --partition=general        # Partition to submit to
        #SBATCH --qos=standard
        #SBATCH --reservation={res}
        
        # Load necessary modules
        module load wulver site
        ml foss/2024a OpenFOAM
            
        # Setup OpenFOAM tutorial    
        source $FOAM_BASH
        cp -R $FOAM_TUTORIALS/incompressible/simpleFoam/{dir} {dir}
        cp OFTOOLS {dir}
        cd {dir}
        mkdir -p constant/triSurface
        cp $FOAM_TUTORIALS/resources/geometry/motorBike.obj.gz constant/triSurface/
        #------------------------------------------------------------------------------------
        . OFTOOLS
        NP=$SLURM_NTASKS
        MPIRUN="mpirun -np $SLURM_NTASKS"
        DCOMP
        ENDTIME 1 1 
        sed "s/200/50/" -i system/controlDict
        sed "s/(20 8 8)/(120 50 50)/"                           -i system/blockMeshDict ;# 34M Mesh
        sed "s/maxLocalCells 100000/maxLocalCells 1000001/"     -i system/snappyHexMeshDict
        sed "s/maxGlobalCells 2000000/maxGlobalCells 20000002/" -i system/snappyHexMeshDict
        ln -s 0.orig 0
        #------------------------------------------------------------------------------------
        # Execute OpenFOAM
        RUNA surfaceFeatureExtract
        RUNA blockMesh
        RUNA decomposePar -force
        RUNP snappyHexMesh -overwrite
        RUNP topoSet
        ls -d processor* | xargs -I {{}} rm -rf ./{{}}/0
        ls -d processor* | xargs -I {{}} cp -r 0.orig ./{{}}/0
        RUNP potentialFoam
        RUNP simpleFoam
        DONE simpleFoam
        
        """
        slurm_script = textwrap.dedent(slurm_script)
        print("Writing the Slurm script")
        
        # Write the SLURM script to a file
        with open("maint_job.slurm", "w") as file:
            file.write(slurm_script)
            print("Slurm script is written\n")
        '''
        # Run the command
        job_submit = submit_job()
        
        job_output = job_submit.splitlines()
        
        print(job_output[0])
        
        # Get the current working directory
        current_directory = os.getcwd()
        print("\nChanging direoctory to:",current_directory)
        
        time.sleep(30)
        
        # Path to your file
        file_path = glob.glob(os.path.join(current_directory, "*.out"))[0]
        print("\nChecking the output file", file_path)
        
        # Check every 5 minutes (300 seconds)
        check_interval = 300
        
        # Get initial modification time
        last_mod_time = os.path.getmtime(file_path)
        
        while True:
                
            # Open and check file content for 'cputime'
            with open(file_path, 'r') as file:
                content = file.read()
                if "ClockTime" in content:
                    print(f"The simulation ran without any errors in {file_path}")
                    print(file_out)
                    break  # Stop the loop if 'cputime' is found
                else:
                    print("The simulation is still in progress\n")
                    out = ['cat', file_path]
                    current_time = datetime.datetime.now().time()
                    print("Printing the output file at\n", current_time)
                    file_out = run_cmd(out)
                    print(file_out)
            
            # Wait for the next check
            time.sleep(check_interval)
        '''
class gromacs:
    def run(self, mode, mig_type=None, reservation=None):
        a100_count, mig_count, l40_count = get_gpu_nodes("gpu")

        if mode == "a100":
            nodes = a100_count
            gres = "gpu:4"
            qos = "standard"
            account = "hpcadmins"
            tasks = "4"
        elif mode == "mig":
            nodes = mig_count
            gres = f"gpu:a100_{mig_type}:4"
            qos = "standard"
            tasks = "4"
            account = "hpcadmins"
        elif mode == "l40":
            nodes = l40_count
            gres = "gpu:l40:2"
            qos = "high_walsh"
            account = "walsh"
            tasks = "2"

        print(f"Running GROMACS in 'gpu' partition ({mode} mode, {nodes} nodes)\n")

        res = resolve_reservation(reservation)
        print("Using Reservation: ", res)

        dir = f'gromacs_{mode}'

        #print("Cleaning up existing files and direcotries\n")

        # Remove all files/directories except "input"
        # del_dir = run_command("find . -mindepth 1 -not -name 'maint_gpu' -exec rm -rf {} +")

        # Create the gromacs directory and change into it
        enter_dir(dir, mig_type if mode == "mig" else None)

        # Build a label used for output filenames (e.g. a100, mig_10g, l40)
        file_label = f"mig_{mig_type}" if mode == "mig" else mode

        # Define the SLURM script with placeholders for variables
        slurm_script = f"""\
        #!/bin/bash
        #SBATCH --job-name=maint_gpu       # Job name
        #SBATCH --array=1-{nodes}
        #SBATCH --account={account}
        #SBATCH --output=gromacs%A_%a.out  # Output file
        #SBATCH --error=gromacs%A_%N_%a.err   # Error log file
        #SBATCH --time=05:00:00            # Wall clock limit (1 hour)
        #SBATCH --nodes=1                  # Number of nodes (dynamic)
        #SBATCH --ntasks-per-node=1
        #SBATCH --cpus-per-task={tasks}
        #SBATCH --partition=gpu            # Partition to submit to
        #SBATCH --qos={qos}
        #SBATCH --reservation={res}
        #SBATCH --gres={gres}
        #SBATCH --exclusive

        module load foss/2025a GROMACS/2025.2-CUDA-12.8.0
        mkdir $SLURM_ARRAY_TASK_ID
        tar -xvzf /project/hpcadmins/am2455/maint/gromacs.tar.gz -C $SLURM_ARRAY_TASK_ID/

        OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
        export GMX_ENABLE_DIRECT_GPU_COMM=true
        export GMX_GPU_DD_COMMS=1
        export GMX_GPU_PME_PP_COMMS=1

        srun --chdir=$SLURM_ARRAY_TASK_ID gmx_mpi mdrun -deffnm equil1 -v -ntomp $SLURM_CPUS_PER_TASK -pin on -tunepme -dlb yes -nb gpu -pme gpu -noappend

        """

        # Dedent the script to remove unnecessary indentation
        slurm_script = textwrap.dedent(slurm_script)

        print("Writing the Slurm script")

        # Write the SLURM script to a file
        with open("maint_job.slurm", "w") as file:
            file.write(slurm_script)
            print("Slurm script is written\n")

        # Run the command
        job_submit = submit_job()

        job_output = job_submit.splitlines()

        print(job_output[0])

        job_id = job_output[0].strip().split()[-1]
        print(f"Waiting for job array {job_id} to complete...")

        # Poll squeue every 60 seconds until all tasks are done
        while True:
            squeue_out = run_command(f"squeue -j {job_id} -h")
            if not squeue_out.strip():
                print(f"Job {job_id} has completed.")
                break
            running = len(squeue_out.strip().splitlines())
            print(f"  {running} task(s) still running, checking again in 60 seconds...")
            time.sleep(60)

        # Parse only this job's .err files (gromacs<jobid>_<node>_<task>.err), not leftovers from earlier runs
        current_directory = os.getcwd()
        file_paths = sorted(glob.glob(os.path.join(current_directory, f"gromacs{job_id}_*.err")))
        print(f"\nParsing {len(file_paths)} output files (expected {nodes})")

        performance_data = []
        failed = []

        for file_path in file_paths:
            with open(file_path, 'r') as file:
                content = file.read()

            fname = os.path.basename(file_path).split('.')[0]
            node_number = fname.split('_')[1]
            file_number = fname.split('_')[-1]

            perf_line = next((line for line in content.split('\n') if "Performance" in line), None)
            if perf_line is None:
                failed.append(file_path)
                print(f"No Performance line found in {file_path} - task did not finish cleanly")
                continue

            parts = perf_line.split()
            performance_ns_day = float(parts[1])
            performance_hour_ns = float(parts[2])
            performance_data.append((file_number, node_number, performance_ns_day, performance_hour_ns))
            print(f"The simulation ran without any errors in {file_path}")

        if failed:
            print(f"\n{len(failed)} of {len(file_paths)} task(s) failed:")
            for file_path in failed:
                print(f"  {file_path}")

        # Sort based on file number
        performance_data.sort(key=lambda x: x[0])

        gmx_performance = f"gmx_performance_{file_label}.csv"
        # Write results to a CSV file
        with open(gmx_performance, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["File Number", "Node Number", "Performance (ns/day)", "Hour/ns"])
            writer.writerows(performance_data)
        # Consolidate results into a table
        gmx_performance_json = f"gmx_performance_{file_label}.json"

        # Convert your performance_data rows into dicts
        records = [
            {
                "file_number": row[0],
                "node": row[1],
                "performance_ns_per_day": row[2],
                "hour_per_ns": row[3],
            }
            for row in performance_data
        ]

        with open(gmx_performance_json, "w") as f:
            json.dump(records, f, indent=2)

        print(f"Performance data has been exported to {gmx_performance}")

class nccl:
    def run(self, mode, mig_type=None, reservation=None):
        a100_count, mig_count, l40_count = get_gpu_nodes("gpu")

        if mode == "a100":
            nodes = a100_count
            gres = "gpu:4"
            qos = "standard"
            account = "hpcadmins"
            tasks = "4"
            gpus = 4
        elif mode == "mig":
            nodes = mig_count
            gres = f"gpu:a100_{mig_type}:4"
            qos = "standard"
            tasks = "4"
            account = "hpcadmins"
            gpus = 4
        elif mode == "l40":
            nodes = l40_count
            gres = "gpu:l40:2"
            qos = "high_walsh"
            account = "walsh"
            tasks = "2"
            gpus = 2

        print(f"Running NCCL in 'gpu' partition ({mode} mode, {nodes} nodes)\n")

        res = resolve_reservation(reservation)
        print("Using Reservation: ", res)

        dir = f'nccl_{mode}'

        enter_dir(dir, mig_type if mode == "mig" else None)

        file_label = f"mig_{mig_type}" if mode == "mig" else mode

        slurm_script = f"""\
        #!/bin/bash
        #SBATCH --job-name=maint_nccl       # Job name
        #SBATCH --array=1-{nodes}
        #SBATCH --account={account}
        #SBATCH --output=nccl%A_%a.out     # Output file
        #SBATCH --error=nccl%A_%N_%a.err   # Error log file
        #SBATCH --time=05:00:00            # Wall clock limit (1 hour)
        #SBATCH --nodes=1                  # Number of nodes (dynamic)
        #SBATCH --ntasks-per-node=1
        #SBATCH --cpus-per-task={tasks}
        #SBATCH --partition=gpu            # Partition to submit to
        #SBATCH --qos={qos}
        #SBATCH --reservation={res}
        #SBATCH --gres={gres}
        #SBATCH --exclusive

        module load CUDA GCCcore NCCL
        nvidia-smi topo -m
        $HOME/nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g {gpus}

        """

        slurm_script = textwrap.dedent(slurm_script)

        print("Writing the Slurm script")

        with open("maint_job.slurm", "w") as file:
            file.write(slurm_script)
            print("Slurm script is written\n")

        job_submit = submit_job()
        job_output = job_submit.splitlines()
        print(job_output[0])

        job_id = job_output[0].strip().split()[-1]
        print(f"Waiting for job array {job_id} to complete...")

        # Poll squeue every 60 seconds until all tasks are done
        while True:
            squeue_out = run_command(f"squeue -j {job_id} -h")
            if not squeue_out.strip():
                print(f"Job {job_id} has completed.")
                break
            running = len(squeue_out.strip().splitlines())
            print(f"  {running} task(s) still running, checking again in 60 seconds...")
            time.sleep(60)

        # Parse all .out files
        current_directory = os.getcwd()
        file_paths = sorted(glob.glob(os.path.join(current_directory, "nccl*.out")))
        print(f"\nParsing {len(file_paths)} output files")

        avg_bw_re = re.compile(r"Avg bus bandwidth\s*:\s*([0-9.+\-Ee]+)")
        records = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                content = f.read()

            fname = os.path.basename(file_path).replace('.out', '')
            task_id = fname.split('_')[-1]

            entries = []
            avg_bus_bw = None
            for line in content.split('\n'):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith('#'):
                    m = avg_bw_re.search(stripped)
                    if m:
                        avg_bus_bw = float(m.group(1))
                    continue
                parts = stripped.split()
                # Data rows: size count type redop root  time algbw busbw #wrong  time algbw busbw #wrong
                if len(parts) < 13:
                    continue
                try:
                    size = int(parts[0])
                    count = int(parts[1])
                except ValueError:
                    continue
                entries.append({
                    "size_bytes": size,
                    "count": count,
                    "type": parts[2],
                    "redop": parts[3],
                    "root": parts[4],
                    "out_of_place": {
                        "time_us": float(parts[5]),
                        "algbw_gbps": float(parts[6]),
                        "busbw_gbps": float(parts[7]),
                        "wrong": parts[8],
                    },
                    "in_place": {
                        "time_us": float(parts[9]),
                        "algbw_gbps": float(parts[10]),
                        "busbw_gbps": float(parts[11]),
                        "wrong": parts[12],
                    },
                })

            records.append({
                "task_id": task_id,
                "avg_bus_bandwidth_gbps": avg_bus_bw,
                "results": entries[-2:],
            })

        out_json = f"nccl_performance_{file_label}.json"
        with open(out_json, "w") as f:
            json.dump(records, f, indent=2)

        print(f"\nWrote {len(records)} records to {out_json}")


class torchcomm:
    def run(self, mode, mig_type=None, reservation=None):
        a100_count, mig_count, l40_count = get_gpu_nodes("gpu")

        if mode == "a100":
            nodes = a100_count
            gres = "gpu:4"
            qos = "standard"
            account = "hpcadmins"
            tasks = "4"
        elif mode == "mig":
            nodes = mig_count
            gres = f"gpu:a100_{mig_type}:4"
            qos = "standard"
            tasks = "4"
            account = "hpcadmins"
        elif mode == "l40":
            nodes = l40_count
            gres = "gpu:l40:2"
            qos = "high_walsh"
            account = "walsh"
            tasks = "2"

        print(f"Running torchcomm in 'gpu' partition ({mode} mode, {nodes} nodes)\n")

        res = resolve_reservation(reservation)
        print("Using Reservation: ", res)

        dir = f'torch_{mode}'

        enter_dir(dir, mig_type if mode == "mig" else None)

        slurm_script = f"""\
        #!/bin/bash
        #SBATCH --job-name=maint_torch      # Job name
        #SBATCH --array=1-{nodes}
        #SBATCH --account={account}
        #SBATCH --output=torch%A_%N_%a.out     # Output file
        #SBATCH --error=torch%A_%N_%a.err   # Error log file
        #SBATCH --time=05:00:00            # Wall clock limit (1 hour)
        #SBATCH --nodes=1                  # Number of nodes (dynamic)
        #SBATCH --ntasks-per-node=1
        #SBATCH --cpus-per-task={tasks}
        #SBATCH --partition=gpu            # Partition to submit to
        #SBATCH --qos={qos}
        #SBATCH --reservation={res}
        #SBATCH --gres={gres}
        #SBATCH --exclusive

        module load Miniforge3
        conda activate torch-cuda
        python $HOME/torch-comm.py

        """

        slurm_script = textwrap.dedent(slurm_script)

        print("Writing the Slurm script")

        with open("maint_job.slurm", "w") as file:
            file.write(slurm_script)
            print("Slurm script is written\n")

        job_submit = submit_job()
        job_output = job_submit.splitlines()
        print(job_output[0])

        job_id = job_output[0].strip().split()[-1]
        print(f"Waiting for job array {job_id} to complete...")

        # Poll squeue every 60 seconds until all tasks are done
        while True:
            squeue_out = run_command(f"squeue -j {job_id} -h")
            if not squeue_out.strip():
                print(f"Job {job_id} has completed.")
                break
            running = len(squeue_out.strip().splitlines())
            print(f"  {running} task(s) still running, checking again in 60 seconds...")
            time.sleep(60)


class hpcg:
    def run(self, reservation=None):
        
        print("Running hpcg test in 'general' partition\n")
        nodes = get_nodes("general")
                                                                                                                                                                            
        res = resolve_reservation(reservation)
        print("Using Reservation: ", res)

        dir = 'hpcg'

        # Create the hpcg directory and change into it
        enter_dir(dir)

        print("Cleaning up existing files and direcotries\n")

        
        del_dir = run_command("find . -mindepth 1 ! -name '*slurm*' ! -name '*.py' -exec rm -rf {} +")
        
        # Define the SLURM script with placeholders for variables                                                                                                           
        slurm_script = f"""\
        #!/bin/bash
        #SBATCH --job-name=maint_hpcg      # Job name
        #SBATCH --output=hpcg-%A_%a.out         # Output file                                                                                                                    
        #SBATCH --error=hpcg-%A_%a.err          # Error log file
        #SBATCH --array=1-{nodes}
        #SBATCH --time=05:00:00            # Wall clock limit (1 hour)                                                                                                      
        #SBATCH --nodes=1            # Number of nodes (dynamic)                                                                                                      
        #SBATCH --ntasks-per-node=128      # Number of tasks (processes)                                                                                                    
        #SBATCH --partition=general        # Partition to submit to                                                                                                         
        #SBATCH --qos=standard                                                                                                                                              
        #SBATCH --reservation={res}
        #SBATCH --exclusive
                                                                                                                                                                            
        # Load necessary modules                                                                                                                                            
        module load wulver
        ml foss/2025a
        DIR=${{SLURM_NODELIST}}_${{SLURM_ARRAY_TASK_ID}}
        mkdir $DIR
        tar -xvzf /project/hpcadmins/am2455/maint/hpcg.tar.gz -C $DIR/
        chmod +x $DIR/xhpcg

        srun --chdir=$DIR ./xhpcg
        """

        # Dedent the script to remove unnecessary indentation                                                                                                               
        slurm_script = textwrap.dedent(slurm_script)

        print("Writing the Slurm script")

        # Write the SLURM script to a file                                                                                                                                  
        with open("maint_job.slurm", "w") as file:
            file.write(slurm_script)
            print("Slurm script is written\n")

        # Run the command                                                                                                                                                   
        job_submit = submit_job()

        job_output = job_submit.splitlines()
        print(job_output[0])

        # Extract job ID from "Submitted batch job 855125"
        job_id = job_output[0].strip().split()[-1]
        print(f"Waiting for job array {job_id} to complete...")

        # Poll squeue every 60 seconds until all tasks are done
        while True:
            squeue_out = run_command(f"squeue -j {job_id} -h")
            if not squeue_out.strip():
                print(f"Job {job_id} has completed.")
                break
            running = len(squeue_out.strip().splitlines())
            print(f"  {running} task(s) still running, checking again in 60 seconds...")
            time.sleep(60)

        gflops_re = re.compile(r"GFLOP/s rating of\s*=\s*([0-9.+\-Ee]+)")
        current_directory = os.getcwd()
        results = []
        failed = []

        # Each array task ran in its own ${SLURM_NODELIST}_${SLURM_ARRAY_TASK_ID} directory
        node_dirs = [entry for entry in sorted(os.listdir(current_directory))
                     if os.path.isdir(os.path.join(current_directory, entry)) and entry.startswith("n")]

        print(f"\nParsing {len(node_dirs)} node directories (expected {nodes})")

        if not node_dirs:
            print("Contents of", current_directory)
            for entry in sorted(os.listdir(current_directory)):
                print(f"  {entry}")
            sys.exit("No per-node directories found - check the .out/.err files above for what the tasks did")

        for entry in node_dirs:
            entry_path = os.path.join(current_directory, entry)
            node_name = entry.split("_")[0]
            task_id = entry.split("_")[-1]

            # The job array has already finished, so the result file is either there or the
            # task failed. Search the whole task directory in case the tarball unpacked
            # into a subdirectory of its own.
            matches = sorted(glob.glob(os.path.join(entry_path, "**", "HPCG-Benchmark*.txt"), recursive=True))

            if not matches:
                failed.append(entry)
                print(f"\n{entry}: no HPCG-Benchmark*.txt found - the run did not finish")
                for suffix in ("err", "out"):
                    log = os.path.join(current_directory, f"hpcg-{job_id}_{task_id}.{suffix}")
                    if os.path.exists(log):
                        tail = run_command(f"tail -n 15 {shlex.quote(log)}")
                        if tail:
                            print(f"  last lines of {os.path.basename(log)}:")
                            for line in tail.splitlines():
                                print(f"    {line}")
                continue

            # Newest result file, in case the directory was reused by an earlier run
            file_path = max(matches, key=os.path.getmtime)

            gflops_value = None
            with open(file_path, "r") as f:
                for line in f:
                    if "HPCG result is VALID" in line and "GFLOP/s rating" in line:
                        m = gflops_re.search(line)
                        if m:
                            gflops_value = float(m.group(1))
                        break

            if gflops_value is None:
                failed.append(entry)
                print(f"{entry}: no VALID GFLOP/s rating in {os.path.basename(file_path)} - benchmark did not validate")
                continue

            results.append(
                {
                    "Node Number": node_name,
                    "GFLOPS": gflops_value,
                }
            )
            print(f"{entry}: GFLOPS = {gflops_value}")

        if failed:
            print(f"\n{len(failed)} of {len(node_dirs)} node(s) produced no result:")
            for entry in failed:
                print(f"  {entry}")

        # Write JSON file with all nodes
        out_json = "hpcg_gflops_results.json"
        with open(out_json, "w") as jf:
            json.dump(results, jf, indent=2)

        print(f"\nWrote {len(results)} records to {out_json}")

                
                
def _parse_bandwidth_table(text):
    """Parse a bandwidth table from nvbandwidth output, handling N/A values."""
    bandwidth = {}
    col_headers = None
    for tline in text.split('\n'):
        stripped = tline.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        # First token must be numeric for this to be a table line
        try:
            float(tokens[0])
        except ValueError:
            continue
        # Parse each token, allowing N/A
        numbers = []
        for t in tokens:
            try:
                numbers.append(float(t))
            except ValueError:
                numbers.append(None)
        # Header row: all values are non-None integers (GPU IDs)
        if col_headers is None and all(n is not None for n in numbers):
            col_headers = [int(n) for n in numbers]
        elif col_headers is not None:
            row_id = int(numbers[0]) if numbers[0] is not None else 0
            row_vals = {}
            for idx, val in enumerate(numbers[1:]):
                if idx < len(col_headers) and val is not None:
                    row_vals[str(col_headers[idx])] = val
            if row_vals:
                bandwidth[str(row_id)] = row_vals
    return bandwidth


class nvlink:
    def run(self, mode, mig_type=None, reservation=None):
        a100_count, mig_count, l40_count = get_gpu_nodes("gpu")

        if mode == "a100":
            nodes = a100_count
            gres = "gpu:4"
            qos = "standard"
            account = "hpcadmins"
            tasks = "4"
        elif mode == "mig":
            nodes = mig_count
            gres = f"gpu:a100_{mig_type}:4"
            qos = "standard"
            account = "hpcadmins"
            tasks = "4"
        elif mode == "l40":
            nodes = l40_count
            gres = "gpu:l40:2"
            qos = "high_walsh"
            account = "walsh"
            tasks = "2"

        print(f"Running NVLink test in 'gpu' partition ({mode} mode, {nodes} nodes)\n")

        res = resolve_reservation(reservation)
        print("Using Reservation: ", res)

        dir = f'nvlink_{mode}'
        enter_dir(dir)

        # Define the SLURM script with placeholders for variables
        slurm_script = f"""\
        #!/bin/bash
        #SBATCH --job-name=maint_nvbandwidth       # Job name
        #SBATCH --array=1-{nodes}
        #SBATCH --account={account}
        #SBATCH --output=nvlink%A_%N_%a.out  # Output file
        #SBATCH --error=nvlink%A_%N_%a.err   # Error log file
        #SBATCH --time=05:00:00            # Wall clock limit (1 hour)
        #SBATCH --nodes=1                  # Number of nodes (dynamic)
        #SBATCH --ntasks-per-node=1
        #SBATCH --cpus-per-task={tasks}
        #SBATCH --partition=gpu            # Partition to submit to
        #SBATCH --qos={qos}
        #SBATCH --reservation={res}
        #SBATCH --gres={gres}
        #SBATCH --exclusive

        module load GCC CUDA
        /home/am2455/nvbandwidth/nvbandwidth
        

        """

        # Dedent the script to remove unnecessary indentation
        slurm_script = textwrap.dedent(slurm_script)

        print("Writing the Slurm script")

        # Write the SLURM script to a file
        with open("maint_job.slurm", "w") as file:
            file.write(slurm_script)
            print("Slurm script is written\n")

        # Run the command
        job_submit = submit_job()

        job_output = job_submit.splitlines()
        print(job_output[0])

        # Extract job ID from "Submitted batch job <id>"
        job_id = job_output[0].strip().split()[-1]
        print(f"Waiting for job array {job_id} to complete...")

        # Poll squeue every 60 seconds until all tasks are done
        while True:
            squeue_out = run_command(f"squeue -j {job_id} -h")
            if not squeue_out.strip():
                print(f"Job {job_id} has completed.")
                break
            running = len(squeue_out.strip().splitlines())
            print(f"  {running} task(s) still running, checking again in 60 seconds...")
            time.sleep(60)

        # Parse all .out files
        current_directory = os.getcwd()
        file_paths = sorted(glob.glob(os.path.join(current_directory, "nvlink*.out")))
        print(f"\nParsing {len(file_paths)} output files")

        records = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                content = f.read()

            # Extract node name from filename (nvlink<jobid>_<nodename>_<taskid>.out)
            fname = os.path.basename(file_path).replace('.out', '')
            parts = fname.split('_')
            node_name = parts[1] if len(parts) >= 3 else fname

            # Parse each test section
            tests = {}
            sections = re.split(r'(?=Running \S+\.)', content)
            for section in sections:
                section = section.strip()
                if not section.startswith("Running "):
                    continue

                # Extract test name
                test_match = re.match(r'Running (\S+)\.\s*\n', section)
                if not test_match:
                    continue
                test_name = test_match.group(1)

                # Find all SUM lines (test_name may have suffixes like _write1, _total)
                all_sums = re.findall(
                    rf'SUM\s+({re.escape(test_name)}\S*)\s+([0-9.+\-Ee]+)', section
                )
                all_cvs = dict(re.findall(
                    rf'COEFFICIENT_OF_VARIATION\s+({re.escape(test_name)}\S*)\s+([0-9.+\-Ee]+)', section
                ))

                if all_sums:
                    test_result = {"status": "pass"}

                    if len(all_sums) == 1 and all_sums[0][0] == test_name:
                        # Simple test — single SUM with exact name match
                        test_result["sum_bandwidth_gbps"] = float(all_sums[0][1])
                        if test_name in all_cvs:
                            test_result["coefficient_of_variation"] = float(all_cvs[test_name])

                        bandwidth = _parse_bandwidth_table(section)
                        if len(bandwidth) == 1:
                            test_result["bandwidth_per_gpu_gbps"] = list(bandwidth.values())[0]
                        elif bandwidth:
                            test_result["bandwidth_matrix_gbps"] = bandwidth
                    else:
                        # Compound test — multiple SUMs with suffixes (e.g. _write1, _write2, _total)
                        subtests = {}
                        sub_parts = re.split(r'(?=memcpy\s+)', section)
                        for sub in sub_parts:
                            sub_sum = re.search(
                                rf'SUM\s+({re.escape(test_name)}\S*)\s+([0-9.+\-Ee]+)', sub
                            )
                            if not sub_sum:
                                continue
                            sum_name = sub_sum.group(1)
                            suffix = sum_name[len(test_name):]
                            if suffix.startswith('_'):
                                suffix = suffix[1:]

                            subtest = {
                                "sum_bandwidth_gbps": float(sub_sum.group(2)),
                            }
                            if sum_name in all_cvs:
                                subtest["coefficient_of_variation"] = float(all_cvs[sum_name])

                            bandwidth = _parse_bandwidth_table(sub)
                            if len(bandwidth) == 1:
                                subtest["bandwidth_per_gpu_gbps"] = list(bandwidth.values())[0]
                            elif bandwidth:
                                subtest["bandwidth_matrix_gbps"] = bandwidth

                            subtests[suffix if suffix else "default"] = subtest

                        test_result["subtests"] = subtests

                    tests[test_name] = test_result
                else:
                    # No SUM line at all means error — capture the error text
                    lines_after = section.split('\n')[1:]
                    error_lines = [l.strip() for l in lines_after if l.strip()]
                    error_msg = '\n'.join(error_lines) if error_lines else "Unknown error"
                    tests[test_name] = {
                        "status": "error",
                        "error": error_msg,
                    }

            records.append({
                "node": node_name,
                "tests": tests,
            })

        # Write JSON
        out_json = "nvlink_results.json"
        with open(out_json, "w") as f:
            json.dump(records, f, indent=2)

        print(f"\nWrote {len(records)} records to {out_json}")

class p2pBandwidthLatencyTest:
    def run(self, mode, mig_type=None, reservation=None):
        a100_count, mig_count, l40_count = get_gpu_nodes("gpu")

        if mode == "a100":
            nodes = a100_count
            gres = "gpu:4"
            qos = "standard"
            account = "hpcadmins"
            tasks = "4"
        elif mode == "mig":
            nodes = mig_count
            gres = f"gpu:a100_{mig_type}:4"
            qos = "standard"
            account = "hpcadmins"
            tasks = "4"
        elif mode == "l40":
            nodes = l40_count
            gres = "gpu:l40:2"
            qos = "high_walsh"
            account = "walsh"
            tasks = "2"

        print(f"Running NVLink test in 'gpu' partition ({mode} mode, {nodes} nodes)\n")

        res = resolve_reservation(reservation)
        print("Using Reservation: ", res)

        dir = f'p2pBandwidthLatencyTest_{mode}'
        enter_dir(dir)

        # Define the SLURM script with placeholders for variables
        slurm_script = f"""\
        #!/bin/bash
        #SBATCH --job-name=maint_p2pBandwidthLatency       # Job name
        #SBATCH --array=1-{nodes}
        #SBATCH --account={account}
        #SBATCH --output=nvlink%A_%N_%a.out  # Output file
        #SBATCH --error=nvlink%A_%N_%a.err   # Error log file
        #SBATCH --time=05:00:00            # Wall clock limit (1 hour)
        #SBATCH --nodes=1                  # Number of nodes (dynamic)
        #SBATCH --ntasks-per-node=1
        #SBATCH --cpus-per-task={tasks}
        #SBATCH --partition=gpu            # Partition to submit to
        #SBATCH --qos={qos}
        #SBATCH --reservation={res}
        #SBATCH --gres={gres}
        #SBATCH --exclusive

        module load GCC CUDA
        /home/am2455/cuda-samples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest
        

        """        
        
        # Dedent the script to remove unnecessary indentation
        slurm_script = textwrap.dedent(slurm_script)

        print("Writing the Slurm script")

        # Write the SLURM script to a file
        with open("maint_job.slurm", "w") as file:
            file.write(slurm_script)
            print("Slurm script is written\n")

        # Run the command
        job_submit = submit_job()

        job_output = job_submit.splitlines()
        print(job_output[0])

        # Extract job ID from "Submitted batch job <id>"
        job_id = job_output[0].strip().split()[-1]
        print(f"Waiting for job array {job_id} to complete...")

        # Poll squeue every 60 seconds until all tasks are done
        # Poll squeue every 60 seconds until all tasks are done
        while True:
            squeue_out = run_command(f"squeue -j {job_id} -h")
            if not squeue_out.strip():
                print(f"Job {job_id} has completed.")
                break
            running = len(squeue_out.strip().splitlines())
            print(f"  {running} task(s) still running, checking again in 60 seconds...")
            time.sleep(60)

        # Parse all .out files
        current_directory = os.getcwd()
        file_paths = sorted(glob.glob(os.path.join(current_directory, "p2pBandwidthLatencyTest*.out")))
        print(f"\nParsing {len(file_paths)} output files")

def _size_to_gib(size):
    """Convert an elbencho size string (4096m, 512g, 1t) to GiB."""
    size = str(size).strip().lower()
    units = {"m": 1.0 / 1024, "g": 1.0, "t": 1024.0, "p": 1024.0 * 1024}

    if size[-1:] in units:
        return int(float(size[:-1]) * units[size[-1]])

    return int(float(size) / (1024 ** 3))  # plain byte count

def _parse_elbencho_resfile(text):
    """Parse an elbencho result file into {operation: {metric: end result}}.

    elbencho prints one block per operation, with two value columns - the result
    when the first thread finished and the end result. We keep the end result.

        WRITE     Elapsed time     :    1m45.698s   1m46.216s
                  IOPS             :          618         617
                  Throughput MiB/s :         2472        2468
    """
    results = {}
    current_op = None

    for line in text.splitlines():
        if ':' not in line:
            continue

        head, _, tail = line.partition(':')
        values = tail.split()
        if not values:
            continue

        if line[:1].isspace():
            # Continuation line - belongs to the operation above
            label = head.strip()
        else:
            parts = head.split()
            if len(parts) < 2 or not parts[0].isupper():
                continue  # header or config line, not a result row
            current_op = parts[0]
            label = " ".join(parts[1:])

        if current_op is None or not label:
            continue

        value = values[-1]
        try:
            value = float(value.replace(',', ''))
        except ValueError:
            pass

        results.setdefault(current_op, {})[label] = value

    return results

class elbencho:
    """Storage/data transfer benchmark.

    single  one array task per node, each node benchmarked against its own
            dataset, so slow nodes stand out against the rest
    multi   one job across every node of a partition, all of them driving a
            single shared dataset through the elbencho service tier
    """

    # Benchmark parameters - tune these

    # Where the datasets are written. $USER is expanded by the job script on the
    # compute node and by the capacity check here, so both agree on the path.
    SCRATCH = "/mmfs1/scratch/hpcadmins/$USER"
    QUOTA = "10t"          # scratch quota - runs are sized to stay inside it

    MODULE = "GCC/13.3.0 elbencho"
    # CUDA build, used for the GPU node tests (-g)
    MODULE_CUDA = "GCC/13.3.0 elbencho/3.0-27-GCC-13.3.0-CUDA-12.6.0"
    THREADS = 16           # I/O threads per node
    BLOCK = "1m"           # block size

    # Dataset written per node. This is below the 512 GB of RAM a node has, so
    # --direct is what keeps the page cache out of the measurement rather than
    # the dataset being too big to cache. Raise it with -s for a run that
    # defeats caching on its size alone (512g is 1x RAM, 1t is 2x).
    #
    # single: every running task holds SIZE, so the quota caps how many run at
    #         once and the array works through the nodes in waves
    # multi:  one shared dataset of nodes x SIZE, capped at the quota
    SIZE = "250g"

    # Published elbencho numbers to sanity check against
    REFERENCE = [
        "single node, 16 threads, 4m block, --direct, 20g: ~212 MiB/s write, 53 IOPS (MSU ICER docs)",
        "single node, 16 threads, 4k random read, --direct: ~7400 IOPS, ~28 MiB/s (MSU ICER docs)",
        "32 nodes x 16 processes: 37.14 GiB/s aggregate, 91% of peak (GWDG, SC23)",
        "Lustre scaling: ~6 GiB/s write and ~9 GiB/s read per storage server node",
        "GDS vs host bounce buffer on direct attached NVMe: ~24 GB/s vs ~6 GB/s read (NVIDIA)",
        "DGX A100 with 8x200Gb IB to NFS storage: 162 GiB/s peak read with GDS (VAST)",
    ]

    def run(self, mode, partition=None, reservation=None, size=None, threads=None,
            gpu=None, gds=False):
        self.size = size or self.SIZE
        self.threads = threads or self.THREADS
        self.gpu = gpu
        self.gds = gds
        self.module = self.MODULE

        if gpu:
            # GPU tests always run on the gpu partition, so the third positional
            # carries the MIG profile instead of a partition name
            self.module = self.MODULE_CUDA
            self._set_gpu_layout(gpu, partition)
            partition = "gpu"
        else:
            partition = partition or "general"
            self.gpu_node_count = None

            if not re.fullmatch(r"[A-Za-z0-9_-]+", partition):
                sys.exit(f"'{partition}' is not a valid partition name")

        self._check_module()

        if mode == "single":
            self.run_single(partition, reservation)
        elif mode == "multi":
            self.run_multi(partition, reservation)
        else:
            sys.exit(f"Unknown elbencho mode '{mode}': expected single or multi")

    #-------------------------------------------------------------------------
    # one array task per node
    #-------------------------------------------------------------------------
    def run_single(self, partition, reservation):
        print(f"Running elbencho single node test in '{partition}' partition\n")
        nodes = self._node_count(partition)

        res = resolve_reservation(reservation)
        print("Using Reservation: ", res)

        # Every running task holds its own dataset, so the quota decides how
        # many of them may run at the same time
        self._clear_stale_scratch()

        concurrency = self._concurrency(nodes)
        array = f"1-{nodes}" if concurrency >= nodes else f"1-{nodes}%{concurrency}"

        self._check_scratch_space("single", _size_to_gib(self.size) * concurrency, nodes)

        enter_dir(f'elbencho_single_{partition}', clean=True)

        slurm_script = f"""\
        #!/bin/bash
        #SBATCH --job-name=maint_elbencho_single   # Job name
        #SBATCH --output=elbencho-single-%A_%a.out # Output file
        #SBATCH --error=elbencho-single-%A_%a.err  # Error log file
        #SBATCH --array={array}
        #SBATCH --time=05:00:00            # Wall clock limit
        #SBATCH --nodes=1                  # One node per array task
        #SBATCH --ntasks-per-node=1
        #SBATCH --cpus-per-task={self.threads}
        #SBATCH --partition={partition}    # Partition to submit to
        {self._qos_directives()}
        #SBATCH --reservation={res}
        #SBATCH --exclusive

        # Load necessary modules
        module load wulver
        ml {self.module}

        NODE=$(hostname -s)
        SCRATCH_DIR={self.SCRATCH}
        DATA_DIR=$SCRATCH_DIR/elbencho/single_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
        RESFILE=elbencho-single_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}_${{NODE}}.txt

        mkdir -p $DATA_DIR

        # Scratch is quota limited, so the dataset goes away whatever happens -
        # a clean finish, a failure, scancel or the wall clock running out
        cleanup() {{
                rm -rf "$DATA_DIR"
        }}
        trap cleanup EXIT TERM INT

        # Each array task benchmarks its own node against its own dataset
        elbencho \\
                --threads {self.threads} \\
                --size {self.size} \\
                --block {self.BLOCK} \\
                --blockvaralgo fast \\
                --blockvarpct 100 \\
                {self._io_flags()} \\
                --write \\
                --read \\
                --delfiles \\
                --resfile $RESFILE \\
                $DATA_DIR/elbencho_${{SLURM_ARRAY_TASK_ID}}.data
        """

        job_id = self._submit(slurm_script, nodes)

        # One result file per array task, named with the node it ran on
        current_directory = os.getcwd()
        res_files = sorted(glob.glob(os.path.join(current_directory, f"elbencho-single_{job_id}_*.txt")))
        print(f"\nParsing {len(res_files)} result files (expected {nodes})")

        records = []
        for file_path in res_files:
            with open(file_path, 'r') as f:
                parsed = _parse_elbencho_resfile(f.read())

            if not parsed:
                print(f"{os.path.basename(file_path)}: no results found")
                continue

            # elbencho-single_<jobid>_<task>_<node>.txt
            parts = os.path.basename(file_path).replace('.txt', '').split('_')
            record = {"node": parts[-1], "task_id": parts[-2]}
            record.update(parsed)
            records.append(record)

            print(f"{record['node']}: "
                  f"write {parsed.get('WRITE', {}).get('Throughput MiB/s', 'n/a')} MiB/s, "
                  f"read {parsed.get('READ', {}).get('Throughput MiB/s', 'n/a')} MiB/s")

        self._report_missing(current_directory, job_id, nodes, len(records), "elbencho-single")

        self._cleanup_scratch("single", job_id)

        self._write_json("single", partition, res, job_id, nodes,
                         self.size, f"{_size_to_gib(self.size) * nodes}g", records)

    #-------------------------------------------------------------------------
    # one job across every node of the partition
    #-------------------------------------------------------------------------
    def run_multi(self, partition, reservation):
        print(f"Running elbencho multi node test across the '{partition}' partition\n")
        nodes = self._node_count(partition)

        res = resolve_reservation(reservation)
        print("Using Reservation: ", res)

        self._clear_stale_scratch()

        # One shared dataset, sized so every node still writes self.size, unless
        # that would not fit in the scratch quota
        total_gib = _size_to_gib(self.size) * nodes
        usable_gib = self._usable_gib()

        if total_gib > usable_gib:
            print(f"\n{self.size} per node across {nodes} nodes needs {total_gib} GiB, "
                  f"more than the {self.QUOTA} scratch quota allows")
            total_gib = usable_gib

        total_size = f"{total_gib}g"
        print(f"Shared dataset: {total_size} ({round(total_gib / nodes, 1)} GiB per node "
              f"across {nodes} nodes)")

        self._check_scratch_space("multi", total_gib, nodes)

        enter_dir(f'elbencho_multi_{partition}', clean=True)

        slurm_script = f"""\
        #!/bin/bash
        #SBATCH --job-name=maint_elbencho_multi  # Job name
        #SBATCH --output=elbencho-multi-%j.out   # Output file
        #SBATCH --error=elbencho-multi-%j.err    # Error log file
        #SBATCH --time=05:00:00            # Wall clock limit
        #SBATCH --nodes={nodes}            # Every node in the partition, one job
        #SBATCH --ntasks-per-node=1
        #SBATCH --cpus-per-task={self.threads}
        #SBATCH --partition={partition}    # Partition to submit to
        {self._qos_directives()}
        #SBATCH --reservation={res}
        #SBATCH --exclusive

        # Load necessary modules
        module load wulver
        ml {self.module}

        SCRATCH_DIR={self.SCRATCH}
        DATA_DIR=$SCRATCH_DIR/elbencho/multi_${{SLURM_JOB_ID}}
        RESFILE=elbencho-multi_${{SLURM_JOB_ID}}.txt

        mkdir -p $DATA_DIR

        # Build elbencho compatible list of hostnames
        HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr "\\n" ",")
        echo "Running on $SLURM_NNODES nodes: $HOSTNAMES"

        # Scratch is quota limited, so the dataset goes away whatever happens -
        # a clean finish, a failure, scancel or the wall clock running out
        cleanup() {{
                elbencho --hosts "$HOSTNAMES" --quit >/dev/null 2>&1
                rm -rf "$DATA_DIR"
        }}
        trap cleanup EXIT TERM INT

        # Start the service on all nodes - the service has to stay in the
        # foreground, otherwise srun falls through thinking the task is clean
        # and kills the orphaned processes.
        srun --cpu-bind=none elbencho --service --foreground &

        # One shared dataset driven by every node at once. --svcwait gives the
        # services time to open their ports before the benchmark connects.
        elbencho \\
                --hosts "$HOSTNAMES" \\
                --svcwait 60 \\
                --threads {self.threads} \\
                --size {total_size} \\
                --block {self.BLOCK} \\
                --blockvaralgo fast \\
                --blockvarpct 100 \\
                {self._io_flags()} \\
                --write \\
                --read \\
                --delfiles \\
                --resfile $RESFILE \\
                $DATA_DIR/elbencho_shared.data

        # Shut the services down so the job exits cleanly
        elbencho --hosts "$HOSTNAMES" --quit
        """

        job_id = self._submit(slurm_script, nodes)

        current_directory = os.getcwd()
        res_file = os.path.join(current_directory, f"elbencho-multi_{job_id}.txt")

        if not os.path.exists(res_file):
            print(f"\nNo result file {os.path.basename(res_file)} - the run did not finish")
            self._print_log_tail(os.path.join(current_directory, f"elbencho-multi-{job_id}.err"))
            self._print_log_tail(os.path.join(current_directory, f"elbencho-multi-{job_id}.out"))
            self._cleanup_scratch("multi", job_id)
            sys.exit("elbencho multi produced no results")

        with open(res_file, 'r') as f:
            parsed = _parse_elbencho_resfile(f.read())

        print(f"\nAggregate across {nodes} nodes:")
        for operation, metrics in parsed.items():
            print(f"  {operation}")
            for label, value in metrics.items():
                print(f"    {label}: {value}")

        self._cleanup_scratch("multi", job_id)

        records = [dict({"nodes": nodes}, **parsed)]
        self._write_json("multi", partition, res, job_id, nodes, self.size, total_size, records)

    #-------------------------------------------------------------------------
    # shared helpers
    #-------------------------------------------------------------------------
    def _submit(self, slurm_script, nodes):
        """Write the script, submit it, and wait for the job to finish."""
        slurm_script = textwrap.dedent(slurm_script)

        print("Writing the Slurm script")

        with open("maint_job.slurm", "w") as file:
            file.write(slurm_script)
            print("Slurm script is written\n")

        job_submit = submit_job()
        job_output = job_submit.splitlines()
        print(job_output[0])

        job_id = job_output[0].strip().split()[-1]
        print(f"Waiting for job {job_id} to complete...")

        # Poll squeue every 60 seconds until the job is done
        while True:
            squeue_out = run_command(f"squeue -j {job_id} -h")
            if not squeue_out.strip():
                print(f"Job {job_id} has completed.")
                break
            running = len(squeue_out.strip().splitlines())
            print(f"  {running} task(s) still running, checking again in 60 seconds...")
            time.sleep(60)

        return job_id

    def _set_gpu_layout(self, gpu, mig_type):
        """Node count, gres, qos and GPU ids for the GPU node type under test."""
        a100_count, mig_count, l40_count = get_gpu_nodes("gpu")

        if gpu == "a100":
            self.gpu_node_count = a100_count
            self.gres = "gpu:4"
            self.qos = "standard"
            self.account = "hpcadmins"
            self.gpuids = "0,1,2,3"
        elif gpu == "mig":
            if mig_type not in MIG_TYPES:
                sys.exit("elbencho -g mig requires a mig_type: 10g, 20g, or 40g\n"
                         "  Usage: python maint_main.py elbencho single -g mig 20g")
            self.gpu_node_count = mig_count
            self.gres = f"gpu:a100_{mig_type}:4"
            self.qos = "standard"
            self.account = "hpcadmins"
            self.gpuids = "0,1,2,3"
        elif gpu == "l40":
            self.gpu_node_count = l40_count
            self.gres = "gpu:l40:2"
            self.qos = "high_walsh"
            self.account = "walsh"
            self.gpuids = "0,1"
        else:
            sys.exit(f"Unknown GPU type '{gpu}': expected a100, mig, or l40")

        if self.gpu_node_count == 0:
            sys.exit(f'No {gpu} nodes available in the "gpu" partition for testing at the moment.')

        path = "GPUDirect Storage (--gds)" if self.gds else "CUDA host buffers (--cuhostbufreg)"
        print(f"\nGPU test: {gpu}, {self.gpu_node_count} nodes, GPU ids {self.gpuids}, {path}")

    def _check_module(self):
        """Check the module loads and report the features it was built with.

        Modules are not always usable from the login node, so a module that will
        not load here is only a warning - the job still runs it on the compute
        node. A module that does load and reports no CUDA is a real error.
        """
        print(f"\nelbencho module: {self.module}")

        # A login shell so that "module"/"ml" are defined
        version = run_command(
            f"bash -lc {shlex.quote(f'module load wulver && ml {self.module} && elbencho --version')} 2>&1 | head -n 20")

        if not version.strip() or "not found" in version.lower() or "error" in version.lower():
            print(f"  could not load it here to check, continuing")
            if version.strip():
                for line in version.splitlines()[:3]:
                    print(f"    {line.strip()}")
            return

        # --version lists the optional features the module was compiled with
        for line in version.splitlines():
            if any(key in line.lower() for key in ("version", "cuda", "cufile", "gds")):
                print(f"  {line.strip()}")

        if self.gpu and not any(key in version.lower() for key in ("cuda", "cufile", "gds")):
            sys.exit(f"{self.module} reports no CUDA support.\n"
                     f"  Set MODULE_CUDA to the CUDA build, or drop -g to use the plain module")

    def _node_count(self, partition):
        return self.gpu_node_count if self.gpu else get_nodes(partition)

    def _qos_directives(self):
        """The qos, account and gres lines, which only the GPU runs need."""
        if not self.gpu:
            return "#SBATCH --qos=standard"

        return (f"#SBATCH --qos={self.qos}\n"
                f"        #SBATCH --account={self.account}\n"
                f"        #SBATCH --gres={self.gres}")

    def _io_flags(self):
        """How the data gets from the filesystem into memory."""
        if not self.gpu:
            flags = ["--direct"]
        elif self.gds:
            # --gds is shorthand for --direct --cufile --gdsbufreg
            flags = [f'--gpuids "{self.gpuids}"', "--gds"]
        else:
            # Read into GPU memory through a pinned host bounce buffer
            flags = [f'--gpuids "{self.gpuids}"', "--cuhostbufreg", "--direct"]

        return (" \\\n" + " " * 16).join(flags)

    def _usable_gib(self):
        """Scratch the run may use, with headroom left under the quota."""
        return int(_size_to_gib(self.QUOTA) * 0.9)

    def _concurrency(self, nodes):
        """How many array tasks may hold a dataset at once within the quota."""
        allowed = max(1, self._usable_gib() // _size_to_gib(self.size))
        concurrency = min(nodes, allowed)

        if concurrency < nodes:
            waves = (nodes + concurrency - 1) // concurrency
            print(f"\nScratch quota {self.QUOTA} allows {concurrency} of {nodes} tasks at a time "
                  f"({waves} waves of {self.size})")

        return concurrency

    def _clear_stale_scratch(self):
        """Drop datasets left by earlier runs, but not by a job still queued.

        A cancelled job normally cleans up through its trap, but one that was
        killed outright leaves its dataset behind, and that still counts against
        the quota for the next run.
        """
        base = os.path.join(os.path.expandvars(self.SCRATCH), "elbencho")
        leftovers = sorted(glob.glob(os.path.join(base, "*_*")))

        if not leftovers:
            return

        print(f"\n{len(leftovers)} dataset directories left in {base}")

        for path in leftovers:
            name = os.path.basename(path)
            parts = name.split('_')
            job_id = parts[1] if len(parts) > 1 else None

            if job_id and run_command(f"squeue -j {job_id} -h 2>/dev/null").strip():
                print(f"  keeping {name}, job {job_id} is still queued")
                continue

            print(f"  removing {name}")
            run_command(f"rm -rf {shlex.quote(path)}")

    def _cleanup_scratch(self, mode, job_id):
        """Remove this run's datasets, in case a task died before its trap ran."""
        scratch = os.path.expandvars(self.SCRATCH)
        base = os.path.join(scratch, "elbencho")
        leftovers = glob.glob(os.path.join(base, f"{mode}_{job_id}*"))

        if leftovers:
            print(f"\nRemoving {len(leftovers)} leftover dataset directories from scratch")
            run_command("rm -rf " + " ".join(shlex.quote(path) for path in leftovers))

        remaining = run_command(f"du -sh {shlex.quote(base)} 2>/dev/null")
        if remaining.strip():
            print(f"Scratch now holds: {remaining.strip()}")

    def _check_scratch_space(self, mode, required_gib, nodes):
        """Stop before submitting if scratch cannot hold the dataset."""
        scratch = os.path.expandvars(self.SCRATCH)
        required_tib = round(required_gib / 1024, 1)

        print(f"\nDataset needs {required_gib} GiB ({required_tib} TiB) of scratch in {scratch} "
              f"(quota {self.QUOTA})")

        # Anything left behind by an earlier run counts against the same quota
        existing_gib = 0
        existing = run_command(f"du -sBG {shlex.quote(os.path.join(scratch, 'elbencho'))} 2>/dev/null")
        if existing.strip():
            print(f"Scratch already holds: {existing.strip()}")
            try:
                existing_gib = int(existing.split()[0].rstrip('G'))
            except (IndexError, ValueError):
                pass

        usable_gib = self._usable_gib()
        if existing_gib + required_gib > usable_gib:
            sys.exit(f"{existing_gib} GiB of old datasets plus {required_gib} GiB for this run "
                     f"exceeds the {self.QUOTA} quota.\n"
                     f"  Clear the leftovers first: "
                     f"rm -rf {os.path.join(scratch, 'elbencho')}/*")

        # df reports the filesystem, which on a shared scratch can be much larger
        # than the per user quota, so this catches the obvious case only
        avail = run_command(f"df -BG --output=avail {shlex.quote(scratch)} 2>&1 | tail -n 1")
        try:
            avail_gib = int(avail.strip().rstrip('G'))
        except (AttributeError, ValueError):
            print(f"Could not read free space from {scratch}, continuing anyway")
            if avail.strip():
                print(f"  df said: {avail.strip()}")
            if not os.path.isdir(scratch):
                print(f"  {scratch} does not exist - check SCRATCH in the elbencho class")
            return

        print(f"Scratch has {avail_gib} GiB ({round(avail_gib / 1024, 1)} TiB) free")

        if required_gib > avail_gib:
            sys.exit(f"Not enough scratch space: need {required_gib} GiB, {avail_gib} GiB free.\n"
                     f"  Use a smaller dataset, e.g. python maint_main.py elbencho {mode} "
                     f"-s {max(1, avail_gib // (nodes * 2))}g")

    def _print_log_tail(self, log_path, lines=15):
        if not os.path.exists(log_path):
            return
        tail = run_command(f"tail -n {lines} {shlex.quote(log_path)}")
        if tail:
            print(f"  last lines of {os.path.basename(log_path)}:")
            for line in tail.splitlines():
                print(f"    {line}")

    def _report_missing(self, current_directory, job_id, expected, found, prefix, show_logs=5):
        """Point at the logs of the array tasks that produced no result file."""
        if found >= expected:
            return

        missing = [task_id for task_id in range(1, expected + 1)
                   if not glob.glob(os.path.join(current_directory, f"elbencho-single_{job_id}_{task_id}_*.txt"))]

        print(f"\n{len(missing)} of {expected} task(s) produced no result file: "
              + ", ".join(str(task_id) for task_id in missing))

        for task_id in missing[:show_logs]:
            print(f"\n  task {task_id}:")
            self._print_log_tail(os.path.join(current_directory, f"{prefix}-{job_id}_{task_id}.err"))
            self._print_log_tail(os.path.join(current_directory, f"{prefix}-{job_id}_{task_id}.out"))

        if len(missing) > show_logs:
            print(f"\n  ... {len(missing) - show_logs} more, see {prefix}-{job_id}_<task>.err")

    def _write_json(self, mode, partition, res, job_id, nodes, size, total_size, records):
        """Write the run parameters, the per-node results and the reference numbers."""
        throughputs = {}
        for operation in ("WRITE", "READ"):
            values = [record[operation]["Throughput MiB/s"] for record in records
                      if isinstance(record.get(operation), dict)
                      and isinstance(record[operation].get("Throughput MiB/s"), float)]
            if values:
                throughputs[operation] = {
                    "min_mib_s": min(values),
                    "max_mib_s": max(values),
                    "mean_mib_s": round(sum(values) / len(values), 2),
                    "total_mib_s": round(sum(values), 2) if mode == "single" else max(values),
                }

        summary = {
            "test": "elbencho",
            "mode": mode,
            "partition": partition,
            "reservation": res,
            "job_id": job_id,
            "nodes": nodes,
            "parameters": {
                "module": self.module,
                "gpu": self.gpu,
                "gpuids": self.gpuids if self.gpu else None,
                "gpudirect_storage": bool(self.gpu and self.gds),
                "threads_per_node": self.threads,
                "size_per_node": size,
                "size_total": total_size,
                "block": self.BLOCK,
                "scratch_quota": self.QUOTA,
                "blockvaralgo": "fast",
                "blockvarpct": 100,
                "direct": True,
                "operations": ["write", "read", "delete"],
            },
            "throughput_summary": throughputs,
            "reference_results": self.REFERENCE,
            "results": records,
        }

        out_json = f"elbencho_{mode}_{partition}_results.json"
        with open(out_json, "w") as jf:
            json.dump(summary, jf, indent=2)

        print(f"\nWrote {len(records)} records to {out_json}")

        if throughputs:
            print("\nThroughput summary (MiB/s):")
            for operation, stats in throughputs.items():
                print(f"  {operation}: min {stats['min_mib_s']}, max {stats['max_mib_s']}, "
                      f"mean {stats['mean_mib_s']}")

        print("\nPublished elbencho numbers for comparison:")
        for line in self.REFERENCE:
            print(f"  {line}")

def main():
    parser = argparse.ArgumentParser(
        description="Run a specific test from the script.",
        add_help=False  # Disable default help
    )
    parser.add_argument("class_name", nargs="?", choices=["openfoam", "gromacs", "nccl", "torchcomm", "hpcg", "nvlink", "p2pBandwidthLatencyTest", "elbencho"], help=argparse.SUPPRESS)
    parser.add_argument("sub_option", nargs="?", choices=["a100", "mig", "l40", "single", "multi"], help=argparse.SUPPRESS)
    # mig_type for the GPU tests, partition name for elbencho
    parser.add_argument("mig_type", nargs="?", metavar="mig_type|partition", help=argparse.SUPPRESS)
    parser.add_argument("-r", "--reservation", metavar="NAME", default=None, help=argparse.SUPPRESS)
    parser.add_argument("-s", "--size", metavar="SIZE", default=None, help=argparse.SUPPRESS)
    parser.add_argument("-t", "--threads", metavar="N", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("-g", "--gpu", metavar="TYPE", choices=["a100", "mig", "l40"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gds", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-h", "--help", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.help or not args.class_name:
        print("-h         for Help")
        print("-r NAME    use a custom reservation instead of the earliest one")
        print("           e.g. python maint_main.py hpcg -r my_reservation")
        print("openfoam   for running OpenFOAM")
        print("gromacs    for running GROMACS (requires: a100, mig, or l40)")
        print("           mig requires mig_type: 10g, 20g, or 40g")
        print("           e.g. python maint_main.py gromacs mig 20g")
        print("nccl       for running NCCL all_reduce_perf (requires: a100, mig, or l40)")
        print("           mig requires mig_type: 10g, 20g, or 40g")
        print("           e.g. python maint_main.py nccl mig 20g")
        print("torchcomm  for running torch-comm test (requires: a100, mig, or l40)")
        print("           mig requires mig_type: 10g, 20g, or 40g")
        print("           e.g. python maint_main.py torchcomm mig 20g")
        print("hpcg       for running HPCG")
        print("nvlink     for running NVLink (requires: a100, mig, or l40)")
        print("           mig requires mig_type: 10g, 20g, or 40g")
        print("p2pBandwidthLatencyTest  for running P2P Bandwidth Latency Test (requires: a100, mig, or l40)")
        print("                           mig requires mig_type: 10g, 20g, or 40g")
        print("elbencho   for running the elbencho storage benchmark (requires: single or multi)")
        print("           single runs one array task per node, each node on its own dataset")
        print("           multi  runs one job across every node of a partition, one shared dataset")
        print("           an optional partition name follows the mode (default: general)")
        print("           e.g. python maint_main.py elbencho single")
        print("           e.g. python maint_main.py elbencho multi general")
        print("-s SIZE    dataset written per node by elbencho (default: 250g)")
        print("-t N       I/O threads per node for elbencho (default: 16)")
        print("           e.g. python maint_main.py elbencho single -s 512g -t 32")
        print("-g TYPE    run elbencho on gpu nodes with the CUDA build: a100, mig, or l40")
        print("           reads and writes go through GPU memory instead of host memory")
        print("           mig takes its profile where the partition would go")
        print("           e.g. python maint_main.py elbencho single -g a100")
        print("           e.g. python maint_main.py elbencho single -g mig 20g")
        print("--gds      use GPUDirect Storage instead of a host bounce buffer (needs -g)")
        print("           e.g. python maint_main.py elbencho multi -g a100 --gds")
        return

    if args.class_name == "openfoam":
        instance = openfoam()
        instance.run(reservation=args.reservation)
    elif args.class_name == "gromacs":
        require_gpu_mode("gromacs", args.sub_option, args.mig_type)
        instance = gromacs()
        instance.run(args.sub_option, args.mig_type, reservation=args.reservation)
    elif args.class_name == "nccl":
        require_gpu_mode("nccl", args.sub_option, args.mig_type)
        instance = nccl()
        instance.run(args.sub_option, args.mig_type, reservation=args.reservation)
    elif args.class_name == "torchcomm":
        require_gpu_mode("torchcomm", args.sub_option, args.mig_type)
        instance = torchcomm()
        instance.run(args.sub_option, args.mig_type, reservation=args.reservation)
    elif args.class_name == "hpcg":
        instance = hpcg()
        instance.run(reservation=args.reservation)
    elif args.class_name == "nvlink":
        require_gpu_mode("nvlink", args.sub_option, args.mig_type)
        instance = nvlink()
        instance.run(args.sub_option, args.mig_type, reservation=args.reservation)
    elif args.class_name == "p2pBandwidthLatencyTest":
        require_gpu_mode("p2pBandwidthLatencyTest", args.sub_option, args.mig_type)
        instance = p2pBandwidthLatencyTest()
        instance.run(args.sub_option, args.mig_type, reservation=args.reservation)
    elif args.class_name == "elbencho":
        if args.sub_option not in ("single", "multi"):
            sys.exit("elbencho requires a mode: single or multi\n"
                     "  Usage: python maint_main.py elbencho single|multi [partition]")
        instance = elbencho()
        instance.run(args.sub_option, args.mig_type, reservation=args.reservation,
                     size=args.size, threads=args.threads, gpu=args.gpu, gds=args.gds)

if __name__ == "__main__":
    main()
