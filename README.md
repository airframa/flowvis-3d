# flowvis-3d
Repositiory intended to perform the processing of an input point cloud containing Flow-Vis paint 3D data 

## Setup

To set up this project locally, follow these steps:

### Prerequisites

- Ensure you have Python 3.10.11 or higher installed.
- Install Poetry. [See Poetry's documentation for installation instructions](https://python-poetry.org/docs/#installation).

### Installation

1. **Clone the repository**

`git clone https://github.com/airframa/flowvis-3d.git`
`cd flowvis-3d`


2. **Install dependencies**

- Configure Poetry to create the virtual environment within the project directory:

    `poetry config virtualenvs.in-project true`

- Then, install dependencies with Poetry:

    `poetry install`

This creates a `.venv` directory inside your project and installs all dependencies.

### Running the Application

Activate the virtual environment:

- Windows: `.venv\Scripts\activate`
- Unix/MacOS: `source .venv/bin/activate`
- Start the application by running: `python app_launch.py`


🚧 **Work in Progress**: Application currently under development. Further functionalities will be added soon. 







