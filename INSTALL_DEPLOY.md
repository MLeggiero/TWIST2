# TWIST2 Deployment-Only Installation

Minimal installation for deploying the pre-trained controller in simulation (sim2sim) or on the real G1 robot (sim2real), with optional PICO VR teleoperation. **IsaacGym and training dependencies are not required.**

## 1. Create conda environment

Python 3.10 is recommended. The Python 3.8 constraint in the main README exists only for IsaacGym compatibility, which is not needed for deployment.

```bash
conda create -n twist2_deploy python=3.10 -y
conda activate twist2_deploy
```

## 2. Install the `pose` package

```bash
cd pose && pip install -e . && cd ..
```

## 3. Install pip dependencies

The `numpy==1.23.0` pin in the main README exists for IsaacGym compatibility and can be dropped for deployment.

```bash
pip install torch mujoco mujoco-python-viewer redis[hiredis] onnx onnxruntime-gpu rich tqdm opencv-python matplotlib
pip install pyttsx3        # voice feedback
pip install customtkinter  # GUI interface
pip install "numpy<2" "scipy<1.14"
```

## 4. Set up Redis

```bash
sudo apt update && sudo apt install -y redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

Edit `/etc/redis/redis.conf`:
```
bind 0.0.0.0
protected-mode no
```

Then restart:
```bash
sudo systemctl restart redis-server
```

## 5. (Sim2Real only) Install Unitree SDK

Required only if controlling the physical G1 robot from your laptop.

```bash
git clone https://github.com/YanjieZe/unitree_sdk2.git
cd unitree_sdk2

sudo apt-get install build-essential cmake python3-dev python3-pip pybind11-dev
pip install pybind11 pybind11-stubgen numpy

cd python_binding
export UNITREE_SDK2_PATH=$(pwd)/..
bash build.sh --sdk-path $UNITREE_SDK2_PATH

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
sudo cp build/lib/unitree_interface.cpython-*-linux-gnu.so $SITE_PACKAGES/unitree_interface.so

python -c "import unitree_interface; print('Unitree SDK installed successfully')"
cd ../..
```

## 6. (Teleoperation only) Install GMR and PICO SDK

PICO VR teleoperation requires a separate conda environment.

```bash
conda create -n gmr python=3.10 -y
conda activate gmr

git clone https://github.com/YanjieZe/GMR.git
cd GMR && pip install -e . && cd ..

conda install -c conda-forge libstdcxx-ng -y
pip install redis[hiredis] rich tqdm opencv-python loop-rate-limiters scipy
```

Then install the PICO PC Service SDK following the instructions in the main [README.md](./README.md#installation) (step 6).

## What you can skip

| Component | Why it's not needed |
|---|---|
| IsaacGym | Only used for RL training |
| `legged_gym` package | Only used for RL training |
| `rsl_rl` package | Only used for RL training |
| `numpy==1.23.0` pin | Only needed for IsaacGym compatibility |
| TWIST2 motion dataset | Only used for training (example motions in `assets/example_motions/` are included) |
