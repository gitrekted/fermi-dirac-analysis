⚛️ Fermi-Dirac Distribution & Tanh Approximation
A Python-based analysis of the Fermi-Dirac distribution and a smooth differentiable approximation using the hyperbolic tangent function. This project visualizes the temperature dependence of electron occupancy, computes energy derivatives, fits an analytical model, and validates thermal broadening behavior using regression.

✅ Features
Exact Fermi-Dirac distribution evaluation

First and second derivative plots

Smooth curve fitting using tanh approximation

RMSE-based error analysis

Linear regression of fitted α vs temperature

📦 Requirements
numpy

matplotlib

scipy

Install all dependencies with:

nginx
Copy
Edit
pip install numpy matplotlib scipy
🚀 How to Run
Clone or download this repository.

Navigate to the project folder.

Run the script:

nginx
Copy
Edit
python fermi_dirac_analysis.py
The script will:

Plot Fermi-Dirac distributions at multiple temperatures

Show the first and second derivatives around the Fermi level

Fit tanh models to the distributions and compute RMSE

Perform linear regression to validate α ~ k_B T scaling

📖 Background
The Fermi-Dirac distribution describes the occupancy probability of energy states in a system of fermions at a given temperature. In numerical modeling, a differentiable approximation such as the hyperbolic tangent is often used to simplify computations. This project analyzes how well the tanh approximation captures the behavior of the Fermi-Dirac function and quantifies fitting accuracy.

📁 Project Structure
bash
Copy
Edit
fermi-dirac-analysis/
├── fermi_dirac_analysis.py   # Main analysis script  
├── README.md                 # Project documentation  
├── LICENSE                   # MIT License  
├── requirements.txt          # Optional dependency list  
└── .gitignore                # Git exclusions (optional)
📄 License
This project is released under the MIT License. See LICENSE for details.

🙌 Acknowledgments
Developed as part of an internship research project. Open to feedback and collaboration.