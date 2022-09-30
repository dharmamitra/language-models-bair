# Start ssh-agent to manage our private ssh key since we're ssh'ing into multiple remotes
eval `ssh-agent`
ssh-add


for i in `seq 18 26`
do
    # Get GPU information
    gpu_info=$(ssh bri25yu@a$i.millennium.berkeley.edu nvidia-smi)

    # Clean up GPU info output to just lines with GPU RAM info
    gpu_info=$(echo "${gpu_info}" | grep Default)

    # Get GPU array RAM per GPU
    ram_per_gpu=$(echo "${gpu_info}" | head -n 1 | cut -c47-54 | xargs)

    # Get GPU RAM available for all GPUs
    ram_available=$(echo "${gpu_info}" | cut -c35-43 | tr -d '\n' | xargs)

    # Print output
    echo "a${i} has ${ram_per_gpu} GPU RAM per GPU. Used RAM on GPUs:"
    echo -e "\t$ram_available"

done
