# AI for the Routine, Humans for the Complex: Accuracy-Driven Data Labelling  with Mixed Integer Linear Programming

## Overview
The scarcity of accurately labelled data remains a major challenge in deep learning (DL). Many DL approaches rely on semi-supervised methods, which focus on constructing large datasets that require only a minimal amount of human-labelled data. Since DL training algorithms can tolerate moderate label noise, it has generally been acceptable for the accuracy of labels in large training datasets to fall well short of a perfect 100%. However, when it comes to testing DL models, achieving high label accuracy -- as close to 100% as possible -- is paramount for reliable verification. In this paper, we introduce **OPAL**, a human-assisted labelling method that can be configured to target a desired accuracy level while minimizing the manual effort required for labelling. The main contribution of OPAL is a mixed-integer linear program (MILP) formulation for labelling effort minimization subject to a specified accuracy threshold. Our evaluation, based on more than 1600 experiments performed on seven datasets, shows that OPAL, relying on this MILP formulation, achieves an average near-perfect accuracy of 98.8%, just 1.2% below perfect accuracy, cutting manual labelling by more than half. Further, OPAL significantly outperforms supervised and semi-supervised baselines in labelling accuracy across all seven datasets, with large effect sizes, when all methods are provided with the same manual-labelling budget.

## Getting Started

### Prerequisites
- Docker
- Python 3.7 or higher
- Required Python packages (see requirements.txt if available)

### Using Docker
Build and run the project container with GPU support using the provided script:
```bash
./docker.sh
```
This command builds the Docker image and starts a container with necessary configurations to run experiments.

### Running Experiments Directly
You can also run experiments natively with Python:
```bash
python mh_milp.py
```
This script executes the MILP experiments with pre-configured datasets and parameters.

## Project Structure
- **docker.sh**: Script to build and run the Docker container.
- **mh_milp.py**: Main script to conduct experiments.
- **mh_data.py**: Contains data preparation and sampling methods.
- **mh_models.py**: Contains model definition and fine-tuning methods.
- **mh_optimize.py**: Contains the MILP optimization routine.
- **mh_custom_dataset.py**: Custom dataset implementation for synthetic datasets.
- **mh_download.py**: Script to download and process the synthetic datasets.

## Contributing
Contributions are welcome. Please submit issues and pull requests for improvements.


