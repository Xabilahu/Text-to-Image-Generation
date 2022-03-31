#!/bin/bash

project_path='/home/xabi/Documents/EHU/Master/Courses/DL4NLP/project/'
tmux new "cd ${project_path}; streamlit run webapp.py" ';' split -h "cd ${project_path}; ./run_executor.sh" ';' split 'watch -n 0.1 nvidia-smi'