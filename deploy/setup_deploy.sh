#!/bin/bash
deactivate 2>/dev/null || true
conda activate env_deploy
cd deploy
source /opt/ros/humble/setup.bash
