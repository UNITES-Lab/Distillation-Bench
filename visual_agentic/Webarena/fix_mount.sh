#!/bin/bash
mkdir -p ./.auth_$(hostname)
sudo mount --bind $(pwd)/.auth_$(hostname) $(pwd)/.auth
