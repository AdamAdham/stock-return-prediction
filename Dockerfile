FROM tensorflow/tensorflow:2.16.1-gpu

# Install extra Python packages
RUN pip install --upgrade pip
RUN pip install seaborn matplotlib pandas scikit-learn ipython ipykernel

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace

# Default command: interactive shell (change if you want auto-run)
CMD ["bash"]
