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
            reservation_name = reservation_name.replace("StartTime", "")
            start_time = line.split('=')[2].strip()
            start_time = start_time.replace("EndTime", "")  # Replace T with space for easier comparison

            # Compare the StartTime to find the earliest
            if min_start_time is None or start_time < min_start_time:
                min_start_time = start_time
                min_reservation = reservation_name

    if min_reservation:
        print(f"Earliest reservation: {min_reservation} with start time: {min_start_time}")
    else:
        print("No reservations found")

    return min_reservation

def wait_for_file(pattern, check_interval=300):
    """Waits for a file matching the pattern to appear in the directory."""
    while True:
        try:
            # Try to find the file
            file_path = glob.glob(os.path.join(os.getcwd(), pattern))[0]
            print(f"\nFile found: {file_path}")
            return file_path  # Return the found file path
        except IndexError:
            print("\nFile not found. Retrying in 5 minutes...")
            time.sleep(check_interval)  # Wait for 5 minutes before retrying
            
class openfoam:
    def run(self):
        nodes = get_nodes("general")
        
        res = get_earliest_reservation(run_command("scontrol show res"))
        print("Using Reservation: ", res)
        
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
        job_submit = run_command("sbatch maint_job.slurm")
        
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
    def run(self, mode, mig_type=None):
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

        res = get_earliest_reservation(run_command("scontrol show res"))
        print("Using Reservation: ", res)

        dir = f'gromacs_{mode}'

        #print("Cleaning up existing files and direcotries\n")

        # Remove all files/directories except "input"
        # del_dir = run_command("find . -mindepth 1 -not -name 'maint_gpu' -exec rm -rf {} +")

        # Create the gromacs directory and change into it
        os.makedirs(dir, exist_ok=True)
        os.chdir(dir)
        if mode == "mig":
            os.makedirs(mig_type, exist_ok=True)
            os.chdir(mig_type)
        print(f"Created and changed directory to: {os.getcwd()}\n")

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
        job_submit = run_command("sbatch maint_job.slurm")

        job_output = job_submit.splitlines()

        print(job_output[0])

        # Get the current working directory
        current_directory = os.getcwd()
        #print("\nChanging direoctory to:",current_directory)

        time.sleep(30)

        # Path to your file
        file_paths = glob.glob(os.path.join(current_directory, "*.err"))
        print("\nChecking the output files", file_paths)

        # Check every 5 minutes (300 seconds)
        check_interval = 60

        # Get initial modification time
        last_mod_times = {file_path: os.path.getmtime(file_path) for file_path in file_paths}
        performance_data = []

        while True:
            all_found = True
            for file_path in file_paths:
                with open(file_path, 'r') as file:
                    content = file.read()
                    if "Performance" in content:
                        print(f"The simulation ran without any errors in {file_path}")
                        lines = content.split('\n')
                        for line in lines:
                            if "Performance" in line:
                                parts = line.split()
                                performance_ns_day = float(parts[1])
                                performance_hour_ns = float(parts[2])
                                fname = os.path.basename(file_path).split('.')[0]  
                                node_number = fname.split('_')[1]   
                                file_number = fname.split('_')[-1]  
                                performance_data.append((file_number, node_number, performance_ns_day, performance_hour_ns))
                    else:
                        all_found = False
                        print("The simulation is still in progress\n")                
                        current_time = datetime.datetime.now().time()
                        print("Printing the output file at: ", current_time)
                        
            if all_found:
                break

            # Wait for the next check
            time.sleep(check_interval)

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

class hpcg:
    def run(self):
        
        print("Running hpcg test in 'general' partition\n")
        nodes = get_nodes("general")
                                                                                                                                                                            
        res = get_earliest_reservation(run_command("scontrol show res"))
        print("Using Reservation: ", res)

        dir = 'hpcg'

        print("Cleaning up existing files and direcotries\n")

        
        del_dir = run_command("find . -mindepth 1 ! -name '*slurm*' ! -name '*.py' -exec rm -rf {} +")
        
        # Define the SLURM script with placeholders for variables                                                                                                           
        slurm_script = f"""\
        #!/bin/bash
        #SBATCH --job=maint_hpcg
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
        job_submit = run_command("sbatch maint_job.slurm")

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

        for entry in sorted(os.listdir(current_directory)):
            entry_path = os.path.join(current_directory, entry)

            # Only look at directories starting with "n"
            if not os.path.isdir(entry_path):
                continue
            if not entry.startswith("n"):
                continue

            node_name = entry.split("_")[0]
            # Define the pattern for the file
            file_pattern = os.path.join(entry_path, "HPCG-Benchmark*.txt")

            # Wait for the result file to appear (polls every 5 min)
            file_path = wait_for_file(file_pattern)

            # Wait until HPCG finishes and writes the VALID result line
            gflops_value = None
            while gflops_value is None:
                with open(file_path, "r") as f:
                    for line in f:
                        if "HPCG result is VALID" in line and "GFLOP/s rating" in line:
                            m = gflops_re.search(line)
                            if m:
                                gflops_value = float(m.group(1))
                            break
                if gflops_value is None:
                    print(f"  Waiting for HPCG to finish in {entry}...")
                    time.sleep(300)

            if gflops_value is not None:
                results.append(
                    {
                        "Node Number": node_name,
                        "GFLOPS": gflops_value,
                    }
                )
                print(f"{entry}: GFLOPS = {gflops_value}")
            else:
                print(f"Warning: no GFLOP/s rating found in {file_path}")

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
    def run(self, mode, mig_type=None):
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

        res = get_earliest_reservation(run_command("scontrol show res"))
        print("Using Reservation: ", res)

        dir = f'nvlink_{mode}'
        os.makedirs(dir, exist_ok=True)
        os.chdir(dir)
        print(f"Created and changed directory to: {os.getcwd()}\n")

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
        job_submit = run_command("sbatch maint_job.slurm")

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
    def run(self, mode, mig_type=None):
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

        res = get_earliest_reservation(run_command("scontrol show res"))
        print("Using Reservation: ", res)

        dir = f'p2pBandwidthLatencyTest_{mode}'
        os.makedirs(dir, exist_ok=True)
        os.chdir(dir)
        print(f"Created and changed directory to: {os.getcwd()}\n")

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
        job_submit = run_command("sbatch maint_job.slurm")

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

def main():
    parser = argparse.ArgumentParser(
        description="Run a specific test from the script.",
        add_help=False  # Disable default help
    )
    parser.add_argument("class_name", nargs="?", choices=["openfoam", "gromacs", "hpcg", "nvlink", "p2pBandwidthLatencyTest"], help=argparse.SUPPRESS)
    parser.add_argument("sub_option", nargs="?", choices=["a100", "mig", "l40"], help=argparse.SUPPRESS)
    parser.add_argument("mig_type", nargs="?", choices=["10g", "20g", "40g"], help=argparse.SUPPRESS)
    parser.add_argument("-h", "--help", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.help or not args.class_name:
        print("-h         for Help")
        print("openfoam   for running OpenFOAM")
        print("gromacs    for running GROMACS (requires: a100, mig, or l40)")
        print("           mig requires mig_type: 10g, 20g, or 40g")
        print("           e.g. python maint_main.py gromacs mig 20g")
        print("hpcg       for running HPCG")
        print("nvlink     for running NVLink (requires: a100, mig, or l40)")
        print("           mig requires mig_type: 10g, 20g, or 40g")
        print("p2pBandwidthLatencyTest  for running P2P Bandwidth Latency Test (requires: a100, mig, or l40)")
        print("                           mig requires mig_type: 10g, 20g, or 40g")
        return

    if args.class_name == "openfoam":
        instance = openfoam()
        instance.run()
    elif args.class_name == "gromacs":
        if not args.sub_option:
            sys.exit("gromacs requires a mode: a100, mig, or l40\n  Usage: python maint_main.py gromacs a100|mig|l40")
        if args.sub_option == "mig" and not args.mig_type:
            sys.exit("gromacs mig requires a mig_type: 10g, 20g, or 40g\n  Usage: python maint_main.py gromacs mig 10g|20g|40g")
        instance = gromacs()
        instance.run(args.sub_option, args.mig_type)
    elif args.class_name == "hpcg":
        instance = hpcg()
        instance.run()
    elif args.class_name == "nvlink":
        if not args.sub_option:
            sys.exit("nvlink requires a mode: a100, mig, or l40\n  Usage: python maint_main.py nvlink a100|mig|l40")
        if args.sub_option == "mig" and not args.mig_type:
            sys.exit("nvlink mig requires a mig_type: 10g, 20g, or 40g\n  Usage: python maint_main.py nvlink mig 10g|20g|40g")
        instance = nvlink()
        instance.run(args.sub_option, args.mig_type)
    elif args.class_name == "p2pBandwidthLatencyTest":
        if not args.sub_option:
            sys.exit("p2pBandwidthLatencyTest requires a mode: a100, mig, or l40\n  Usage: python maint_main.py p2pBandwidthLatencyTest a100|mig|l40")
        if args.sub_option == "mig" and not args.mig_type:
            sys.exit("p2pBandwidthLatencyTest mig requires a mig_type: 10g, 20g, or 40g\n  Usage: python maint_main.py p2pBandwidthLatencyTest mig 10g|20g|40g")
        instance = p2pBandwidthLatencyTest()
        instance.run(args.sub_option, args.mig_type)

if __name__ == "__main__":
    main()
