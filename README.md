# RoboTrainer Automatic Assessment

Build the Docker image with the following command:
```bash
./build_docker.sh
```
Run the Docker container with the following command:
```bash
./start_docker.sh
```

## Conda Environment
The default conda environment is `robotrainer_automatic_assessment`.

Create a new conda environment with the following command:
```bash
conda create -n my_new_env python=3.12
```

To change the conda environment, edit the `build_docker.sh` file and set the `CONDA_ENV` variable.
```bash
CONDA_ENV=robotrainer_automatic_assessment
```

## Jupyter Notebook
To start a Jupyter Notebook server, run the following command:
```bash
./start_jupyter_server.sh
```
Go to `http://localhost:8888` in your browser to access the Jupyter Notebook.

## TODO
- [ ] Add GPU support