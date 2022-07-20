We provide here the accompanying code for the paper: Theoretical Understanding of the  Information Flow on Continual Learning Performance
The directories and files for the network checkpoints and the dataset can be found here: https://drive.google.com/drive/folders/1seLm67Movn9_pJWJs8lq2gnpWtqAuAQH?usp=sharing

The code was adapted from the published code for Packnet [1] located here: https://github.com/arunmallya/packnet

The code is split into four directories:
1. Data: Holds all of the tasks' testing and training data for our implemented split-CIFAR-10/100 
2. Checkpoints: Stores network states, we provide the final state for only one run (r002) in order to limit file sizes
3. Logs: Stores the runtime output from experiments. We provide all logs which match the provided checkpoint (r002)
4. Src: Provides the commented Python scripts needed to run the code, along with the bash scripts for running experiments end-to-end

Note: Additionally, we provide an .ipynb file used for setting up and running the associated code. 

For any questions, please feel free to contact Joshua.Andle@maine.edu


[1]. Mallya, Arun, and Svetlana Lazebnik. "Packnet: Adding multiple tasks to a single network by iterative pruning." Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2018.
